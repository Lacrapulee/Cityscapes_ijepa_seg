"""
Évalue un modèle entraîné (checkpoint contenant la tête DPT/simple) sur Cityscapes.
Recharge l'encodeur I-JEPA depuis HuggingFace (jamais sauvegardé, toujours identique).

Usage :
  python test.py --data-root /chemin/vers/cityscapes --checkpoint work_dirs/run1/best.pth --split val
  
  # Avec extraction de features :
  python test.py ... --extract-features --features-out ./features_out --num-samples 10
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF
from PIL import Image

from model import IJepaSegmentationModel
from dataset import CityscapesSegDataset, CITYSCAPES_MEAN as IMAGENET_MEAN, CITYSCAPES_STD as IMAGENET_STD, labelid_to_trainid
from train import compute_miou, update_confusion_matrix, NUM_CLASSES

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


# ===========================================================================
# Extraction de features par hook
# ===========================================================================

class LayerFeatureExtractor:
    """
    Attache des forward hooks sur les couches transformer du backbone ViT
    pour capturer les tokens de sortie à chaque couche demandée.

    I-JEPA n'a pas de token CLS (contrairement à ViT/DINO) : tous les tokens
    capturés sont des patches, remis en grille 2D (H_patch × W_patch) puis
    interpolés à la résolution crop_size.

    Convention d'indexation : `layer_indices` suit la même convention que
    `output_hidden_states` côté HuggingFace (et donc que IJepaEncoder.forward
    dans model.py) — l'indice `idx` désigne la sortie après la couche
    transformer numéro `idx - 1` (0-indexée). Un hook est donc posé sur
    `layers[idx - 1]`, pas `layers[idx]`.
    """

    def __init__(self, model: nn.Module, layer_indices: list[int]):
        self.layer_indices = layer_indices
        self._features: dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        # Réutilise la même logique de résolution que IJepaEncoder (model.py),
        # qui gère déjà les variations de nommage HuggingFace (layer/layers,
        # avec ou sans niveau d'encapsulation .encoder supplémentaire).
        layers = model.encoder._get_transformer_layers()

        for idx in self.layer_indices:
            if idx == 0:
                raise ValueError(
                    "layer_indices=0 correspond à la sortie des patch embeddings "
                    "(avant toute couche transformer) ; non capturable via un hook "
                    "de couche. Utilise idx >= 1."
                )
            hook = layers[idx - 1].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            # Les couches transformer HuggingFace renvoient un tuple
            # (hidden_states, ...) et non directement un tensor.
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Pas de token CLS en I-JEPA : tous les tokens sont des patches.
            self._features[idx] = hidden_states.detach().cpu()
        return hook

    def get_features(self) -> dict[int, torch.Tensor]:
        """Retourne les features capturées lors du dernier forward."""
        return dict(self._features)

    def clear(self):
        self._features.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def tokens_to_heatmap(
    tokens: torch.Tensor,
    crop_size: tuple,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Convertit un tenseur de tokens patch (1, N_patches, D) en heatmap 2D
    en moyennant sur la dimension D, puis en interpolant à crop_size.

    Retourne un array float32 normalisé dans [0, 1] de shape (H, W).
    """
    # tokens : (1, N, D) → moyenne sur D → (1, N)
    mean_activation = tokens[0].mean(dim=-1)  # (N,)

    # Retrouve la grille H_p × W_p
    H_in, W_in = crop_size
    H_p = H_in // patch_size
    W_p = W_in // patch_size
    assert mean_activation.numel() == H_p * W_p, (
        f"Nombre de tokens ({mean_activation.numel()}) ≠ H_p×W_p ({H_p}×{W_p}). "
        f"Vérifie patch_size={patch_size}."
    )

    grid = mean_activation.reshape(1, 1, H_p, W_p).float()

    # Upscale vers crop_size
    heatmap = torch.nn.functional.interpolate(
        grid, size=(H_in, W_in), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    # Normalisation [0, 1]
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def save_feature_maps(
    features: dict[int, torch.Tensor],
    image_tensor: torch.Tensor,
    sample_idx: int,
    out_dir: str,
    crop_size: tuple,
    patch_size: int = 14,
):
    """
    Sauvegarde pour chaque couche :
      - la heatmap seule (.png)
      - la heatmap superposée sur l'image originale (_overlay.png)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Dénormalise l'image pour l'overlay
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_np = (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    for layer_idx, tokens in sorted(features.items()):
        heatmap = tokens_to_heatmap(tokens, crop_size, patch_size)

        # --- Heatmap seule ---
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(heatmap, cmap="inferno")
        ax.set_title(f"Sample {sample_idx:04d} — Layer {layer_idx}")
        ax.axis("off")
        path = os.path.join(out_dir, f"sample{sample_idx:04d}_layer{layer_idx:02d}.png")
        fig.savefig(path, bbox_inches="tight", dpi=100)
        plt.close(fig)

        # --- Overlay ---
        heatmap_colored = cm.inferno(heatmap)[..., :3]   # (H, W, 3), float [0,1]
        overlay = 0.55 * img_np + 0.45 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].imshow(img_np);            axes[0].set_title("Image");   axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="inferno"); axes[1].set_title(f"Layer {layer_idx}"); axes[1].axis("off")
        axes[2].imshow(overlay);           axes[2].set_title("Overlay"); axes[2].axis("off")
        fig.suptitle(f"Sample {sample_idx:04d} — Layer {layer_idx}", fontsize=12)
        path_ov = os.path.join(out_dir, f"sample{sample_idx:04d}_layer{layer_idx:02d}_overlay.png")
        fig.savefig(path_ov, bbox_inches="tight", dpi=100)
        plt.close(fig)

    print(f"  [features] sample {sample_idx:04d} → {out_dir}")


# ===========================================================================
# Inférence fenêtre glissante (inchangé)
# ===========================================================================

@torch.no_grad()
def sliding_window_inference(model, image_tensor, crop_size, stride, num_classes, device):
    _, H, W = image_tensor.shape
    crop_h, crop_w = crop_size
    stride_h, stride_w = stride

    logits_sum = torch.zeros(num_classes, H, W, device=device)
    count      = torch.zeros(1, H, W, device=device)

    y = 0
    while y < H:
        y_end   = min(y + crop_h, H)
        y_start = max(y_end - crop_h, 0)
        x = 0
        while x < W:
            x_end   = min(x + crop_w, W)
            x_start = max(x_end - crop_w, 0)

            window = image_tensor[:, y_start:y_end, x_start:x_end].unsqueeze(0).to(device)
            logits = model(window).squeeze(0)

            logits_sum[:, y_start:y_end, x_start:x_end] += logits
            count[:,      y_start:y_end, x_start:x_end] += 1

            if x_end == W: break
            x += stride_w
        if y_end == H: break
        y += stride_h

    return logits_sum / count.clamp(min=1)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",       required=True)
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--model-name",      default="facebook/ijepa_vith14_22k")
    parser.add_argument("--split",           default="val", choices=["val", "test"])
    parser.add_argument("--crop-size",       type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--layer-indices",   type=int, nargs="+", default=[7, 15, 23, 31])
    parser.add_argument("--decoder-type",    default="dpt",     choices=["dpt", "simple"])
    parser.add_argument("--fusion-type",     default="feature", choices=["feature", "multidepth"])
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--full-res-eval",   action="store_true")
    parser.add_argument("--stride",          type=int, nargs=2, default=None)
    # --- nouveaux args features ---
    parser.add_argument("--extract-features", action="store_true",
                        help="Active l'extraction de feature maps par couche")
    parser.add_argument("--features-out",     default="./feature_maps",
                        help="Dossier de sortie pour les visualisations")
    parser.add_argument("--num-samples",      type=int, default=10,
                        help="Nombre d'images pour lesquelles on sauvegarde les features")
    parser.add_argument("--patch-size",       type=int, default=14,
                        help="Taille des patches du ViT (14 pour ViT-H/14)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = tuple(args.crop_size)   # (H, W)

    model = IJepaSegmentationModel(
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        layer_indices=tuple(args.layer_indices),
        unfreeze_last_n=args.unfreeze_last_n,
        decoder_type=args.decoder_type,
        fusion_type=args.fusion_type,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "head" in ckpt:
        model.head.load_state_dict(ckpt["head"])
        model.load_trainable_backbone_state_dict(ckpt.get("backbone", {}))
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif "fusion" in ckpt and "decoder" in ckpt:
        # Format legacy (anciens checkpoints type run3) : la tête SimpleHead a été
        # sauvegardée comme deux state_dicts séparés "fusion"/"decoder" au lieu
        # d'un seul state_dict "head". Ne correspond qu'à decoder_type="simple".
        if not (hasattr(model.head, "fusion") and hasattr(model.head, "decoder")):
            raise ValueError(
                "Le checkpoint contient des clés 'fusion'/'decoder' (format SimpleHead "
                f"legacy), incompatible avec decoder_type={args.decoder_type!r}. "
                "Relance avec --decoder-type simple (et --fusion-type multidepth pour "
                "ce run)."
            )
        model.head.fusion.load_state_dict(ckpt["fusion"])
        model.head.decoder.load_state_dict(ckpt["decoder"])
    else:
        # Le checkpoint EST directement le state_dict
        model.load_state_dict(ckpt, strict=False)
    print(f"Checkpoint chargé (epoch {ckpt['epoch']}, mIoU {ckpt['miou']:.4f})")
    model.eval()

    # --- Attache les hooks si extraction demandée ---
    extractor = None
    if args.extract_features:
        extractor = LayerFeatureExtractor(model, args.layer_indices)
        print(f"Extraction activée sur les couches {args.layer_indices} "
              f"→ {args.features_out} ({args.num_samples} samples)")

    conf_matrix  = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    sample_count = 0

    dataset = CityscapesSegDataset(
        args.data_root, split=args.split, crop_size=crop_size, train=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)

            if args.full_res_eval:
                stride = (
                    tuple(args.stride) if args.stride
                    else (crop_size[0] * 2 // 3, crop_size[1] * 2 // 3)
                )
                logits = sliding_window_inference(
                    model, images[0], crop_size, stride, NUM_CLASSES, device
                )
            else:
                logits = model(images_dev).squeeze(0)

            preds = logits.argmax(dim=0).cpu().flatten()
            update_confusion_matrix(conf_matrix, preds, labels.flatten(), NUM_CLASSES)

            # --- Sauvegarde des features sur les N premiers samples ---
            if extractor is not None and sample_count < args.num_samples:
                # La taille réelle de l'image envoyée au modèle (donc la grille de
                # patches réellement produite) ne correspond PAS forcément à
                # --crop-size : en val/test, CityscapesSegDataset._preprocess_val
                # ignore crop_size et redimensionne à la résolution native
                # Cityscapes (garde le ratio). On lit donc la taille directement
                # sur le tensor plutôt que de faire confiance à args.crop_size.
                actual_hw = tuple(images.shape[-2:])
                save_feature_maps(
                    features=extractor.get_features(),
                    image_tensor=images[0],       # (C, H, W) non batchifié
                    sample_idx=sample_count,
                    out_dir=args.features_out,
                    crop_size=actual_hw,
                    patch_size=args.patch_size,
                )
            if extractor is not None:
                extractor.clear()

            sample_count += 1

    if extractor is not None:
        extractor.remove_hooks()

    miou = compute_miou(conf_matrix)
    print(f"\nmIoU global ({args.split}) : {miou:.4f}\n")
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(0) + conf_matrix.sum(1) - intersection
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s} : {(intersection[i] / union[i].clamp(min=1)).item():.4f}")


if __name__ == "__main__":
    main()