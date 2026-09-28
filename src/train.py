"""
Entraînement du convertisseur + décodeur sur Cityscapes, encodeur I-JEPA gelé.

Usage :
  python train.py --data-root /chemin/vers/cityscapes --output-dir work_dirs/run1
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    worker_seed = torch.initial_seed() % 2**32
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

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
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
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
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

    n_trainable = sum(p.numel() for g in model.param_groups(args.lr) for p in g["params"])
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {n_trainable / 1e6:.1f}M / {n_total / 1e6:.1f}M total")

    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr, args.backbone_lr_mult), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.head.load_state_dict(ckpt["head"])
        model.load_trainable_backbone_state_dict(ckpt.get("backbone", {}))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        print(f"Reprise depuis {args.resume} : epoch {start_epoch}, meilleur mIoU jusqu'ici {best_miou:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 20 == 0:
                print(f"[epoch {epoch}] step {i}/{len(train_loader)} - loss {loss.item():.4f}")
                if args.wandb:
                    wandb.log({"train/step_loss": loss.item(), "epoch": epoch})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} terminée en {time.time() - t0:.1f}s - loss moyenne {avg_loss:.4f}")
        if args.wandb:
            wandb.log({"train/epoch_loss": avg_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch})

        torch.save(
            {
                "head": model.head.state_dict(),
                "backbone": model.get_trainable_backbone_state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
            },
            os.path.join(args.output_dir, "last.pth"),
        )

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                for images, labels in val_loader:
                    images = images.to(device)
                    logits = model(images)
                    preds = logits.argmax(dim=1).cpu().flatten()
                    update_confusion_matrix(conf_matrix, preds, labels.flatten(), NUM_CLASSES)

            miou = compute_miou(conf_matrix)
            print(f"[val] epoch {epoch} - mIoU: {miou:.4f}")
            if args.wandb:
                wandb.log({"val/miou": miou, "epoch": epoch})

            if miou > best_miou:
                best_miou = miou
                torch.save(
                    {
                        "head": model.head.state_dict(),
                        "backbone": model.get_trainable_backbone_state_dict(),
                        "epoch": epoch,
                        "miou": miou,
                    },
                    os.path.join(args.output_dir, "best.pth"),
                )
                print(f"Nouveau meilleur modèle sauvegardé (mIoU {miou:.4f})")

    print(f"Entraînement terminé. Meilleur mIoU : {best_miou:.4f}")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()