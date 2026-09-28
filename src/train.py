"""
Entraînement du convertisseur + décodeur sur Cityscapes, encodeur I-JEPA gelé.

Multi-GPU / précision mixte via HuggingFace Accelerate. Le même script tourne en
mono-GPU (python) ou en distribué (accelerate launch) :

  # mono-GPU (comportement identique à avant)
  python src/train.py --data-root data/cityscapes --output-dir work_dirs/run1

  # 4 GPU, DDP
  accelerate launch --multi_gpu --num_processes 4 src/train.py --data-root data/cityscapes ...

  # + précision mixte bf16
  accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 src/train.py ...

--batch-size est PAR GPU : batch effectif = batch_size × nb_GPU × gradient_accumulation_steps.
Pour reproduire un run mono-GPU batch 4 sur 4 GPU : --batch-size 1 (ou garder 4 et
adapter le LR ; aucune mise à l'échelle automatique du LR n'est faite).
Les checkpoints gardent exactement le même format (clés head/backbone/...) : test.py
et --resume restent compatibles avec les anciens runs.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from model import IJepaSegmentationModel
from dataset import CityscapesSegDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

NUM_CLASSES = 19


def compute_miou(conf_matrix: torch.Tensor) -> float:
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(0) + conf_matrix.sum(1) - intersection
    iou = intersection / union.clamp(min=1)
    return iou[union > 0].mean().item()


def update_confusion_matrix(conf_matrix, preds, targets, num_classes, ignore_index=255):
    mask = targets != ignore_index
    preds, targets = preds[mask], targets[mask]
    idx = targets * num_classes + preds
    conf_matrix += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def set_seed(seed: int):
    """Seed random/numpy/torch dans le process principal (poids, ordre de shuffle)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """
    À passer en worker_init_fn du DataLoader.

    Chaque worker est un fork du process principal : sans ça, ils héritent tous
    du même état du module `random` (utilisé directement dans dataset.py pour
    le scale-jitter, le crop et la PhotoMetricDistortion) et peuvent produire
    des séquences d'augmentation corrélées entre eux. PyTorch dérive déjà une
    seed différente par worker (base_seed + worker_id) accessible via
    torch.initial_seed() — on s'en sert pour reseed random/numpy séparément.
    """
    # En distribué, le générateur du DataLoader est synchronisé entre processus :
    # sans le rang dans la seed, tous les GPU tireraient la même séquence d'augmentations.
    rank = int(os.environ.get("RANK", 0))
    worker_seed = (torch.initial_seed() + 1_000_003 * rank) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Racine du dataset Cityscapes")
    parser.add_argument("--output-dir", default="work_dirs/run1")
    parser.add_argument("--model-name", default="facebook/ijepa_vith14_22k")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--layer-indices", type=int, nargs=4, default=[7, 15, 23, 31],
                         help="Exactement 4 indices de couches d'I-JEPA, requis par la tête DPT "
                              "(Reassemble/Fusion câblée pour 4 échelles géométriques : x4/x2/x1/x0.5).")
    parser.add_argument("--decoder-type", default="dpt", choices=["dpt", "simple"],
                         help="'dpt' = Reassemble multi-échelle. "
                              "'simple' =  upsampling progressif.")
    parser.add_argument("--fusion-type", default="feature", choices=["feature", "multidepth"],
                         help="'feature' =  Fusion RefineNet. "
                              "'simple' = ancienne fusion multi-profondeur (résolution fixe).")
    parser.add_argument("--unfreeze-last-n", type=int, default=0,
                         help="Nombre de dernières couches d'I-JEPA à dégeler (0 = backbone entièrement gelé)")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                         help="Multiplicateur du LR pour les couches dégelées du backbone (relatif à --lr)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", default=None, choices=["no", "fp16", "bf16"],
                        help="Défaut : valeur de `accelerate launch --mixed_precision` / accelerate config "
                             "(sinon fp32). Les runs existants ont été entraînés en fp32.")
    parser.add_argument("--wandb", action="store_true", help="Activer le logging Weights & Biases")
    parser.add_argument("--wandb-project", default="ijepa-cityscapes-seg")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None,
                         help="Chemin vers last.pth pour reprendre un entraînement interrompu")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed pour random/numpy/torch (+ workers du DataLoader). "
                              "Fixe-la à la même valeur entre deux runs pour comparer des "
                              "architectures sans que le bruit (crops/augmentations/init) "
                              "ne pollue la comparaison.")
    args = parser.parse_args()

    # find_unused_parameters : certains paramètres entraînables ne reçoivent jamais de
    # gradient, ce que DDP refuse par défaut :
    #  - le bloc transformer 32 quand --unfreeze-last-n couvre des couches au-delà de
    #    max(--layer-indices) (la tête ne lit que hidden_states[31]) ;
    #  - resConfUnit1 du premier bloc de fusion DPT (appelé avec une seule entrée).
    # En mono-GPU ces paramètres restent simplement inchangés (AdamW les ignore) : même
    # comportement ici.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device
    is_main = accelerator.is_main_process
    world = accelerator.num_processes
    log = accelerator.print

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    # Même seed sur tous les processus : initialisation de la tête identique
    # (DDP la diffuse de toute façon depuis le rang 0).
    set_seed(args.seed)
    log(f"{world} processus | précision {accelerator.mixed_precision} | batch effectif "
        f"{args.batch_size * world * args.gradient_accumulation_steps}")

    if args.wandb and is_main:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb n'est pas installé -- lancez `pip install wandb` ou retirez --wandb")
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args),
                   resume="allow", id=args.run_name)

    train_set = CityscapesSegDataset(args.data_root, split="train", crop_size=tuple(args.crop_size), train=True)
    val_set = CityscapesSegDataset(args.data_root, split="val", crop_size=tuple(args.crop_size), train=False)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=seed_worker, generator=loader_generator,
    )
    # Validation : partition exacte des images entre processus (i::world), sans passer
    # par accelerator.prepare dont le sampler dupliquerait des images pour égaliser les
    # shards (500 % world != 0) et fausserait la matrice de confusion.
    val_shard = Subset(val_set, range(accelerator.process_index, len(val_set), world))
    val_loader = DataLoader(
        val_shard, batch_size=1, shuffle=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=seed_worker,
    )

    model = IJepaSegmentationModel(
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        layer_indices=tuple(args.layer_indices),
        unfreeze_last_n=args.unfreeze_last_n,
        decoder_type=args.decoder_type,
        fusion_type=args.fusion_type,
    ).to(device)
    if world > 1 and device.type == "cuda":  # SyncBatchNorm n'existe que sur GPU
        # BatchNorm des têtes simple/multidepth : statistiques sur le batch global
        # plutôt que sur les batch_size images de chaque GPU.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_trainable = sum(p.numel() for g in model.param_groups(args.lr) for p in g["params"])
    n_total = sum(p.numel() for p in model.parameters())
    log(f"Paramètres entraînables : {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M total")

    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr, args.backbone_lr_mult), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.head.load_state_dict(ckpt["head"])
        model.load_trainable_backbone_state_dict(ckpt.get("backbone", {}))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        log(f"Reprise depuis {args.resume} : epoch {start_epoch}, meilleur mIoU jusqu'ici {best_miou:.4f}")

    # Le scheduler n'est PAS passé à prepare() : il est steppé une fois par epoch
    # (T_max=epochs), alors qu'Accelerate le ferait avancer num_processes fois par appel.
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    raw_model = accelerator.unwrap_model(model)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (images, labels) in enumerate(train_loader):
            # prepare() a déjà placé le batch sur le bon device
            with accelerator.accumulate(model):
                logits = model(images)
                loss = criterion(logits, labels)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            if i % 20 == 0:
                log(f"[epoch {epoch}] step {i}/{len(train_loader)} - loss {loss.item():.4f}")
                if args.wandb and is_main:
                    wandb.log({"train/step_loss": loss.item(), "epoch": epoch})

        scheduler.step()
        avg_loss = accelerator.reduce(
            torch.tensor(epoch_loss / len(train_loader), device=device), reduction="mean"
        ).item()
        log(f"Epoch {epoch} terminée en {time.time() - t0:.1f}s - loss moyenne {avg_loss:.4f}")
        if args.wandb and is_main:
            wandb.log({"train/epoch_loss": avg_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch})

        if (epoch + 1) % args.val_interval == 0:
            # Modèle "déballé" : pas de collectives DDP dans le forward, donc des shards
            # de tailles différentes (500 % nb_GPU != 0) ne bloquent pas.
            raw_model.eval()
            conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                for images, labels in val_loader:
                    logits = raw_model(images.to(device))
                    preds = logits.argmax(dim=1).flatten()
                    update_confusion_matrix(conf_matrix, preds, labels.to(device).flatten(), NUM_CLASSES)
            conf_matrix = accelerator.reduce(conf_matrix, reduction="sum").cpu()

            miou = compute_miou(conf_matrix)
            log(f"[val] epoch {epoch} - mIoU: {miou:.4f}")
            if args.wandb and is_main:
                wandb.log({"val/miou": miou, "epoch": epoch})

            if miou > best_miou:
                best_miou = miou
                if is_main:
                    torch.save(
                        {
                            "head": raw_model.head.state_dict(),
                            "backbone": raw_model.get_trainable_backbone_state_dict(),
                            "epoch": epoch,
                            "miou": miou,
                        },
                        os.path.join(args.output_dir, "best.pth"),
                    )
                log(f"Nouveau meilleur modèle sauvegardé (mIoU {miou:.4f})")

        # last.pth APRÈS la validation : sinon son best_miou retarde d'une validation et
        # un --resume pourrait écraser un meilleur best.pth par un modèle moins bon.
        accelerator.wait_for_everyone()
        if is_main:
            torch.save(
                {
                    "head": raw_model.head.state_dict(),
                    "backbone": raw_model.get_trainable_backbone_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou,
                },
                os.path.join(args.output_dir, "last.pth"),
            )
        accelerator.wait_for_everyone()

    log(f"Entraînement terminé. Meilleur mIoU : {best_miou:.4f}")
    if args.wandb and is_main:
        wandb.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()