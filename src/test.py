"""
Évaluation comparative des têtes de segmentation I-JEPA (ViT-H/14) sur Cityscapes val.

Modèles comparés (même backbone, mêmes couches 7/15/23/31, 4 derniers blocs dégelés,
crops 512×512, seed 42 — seules la tête et la fusion changent) :

    simple512x512          SimpleHead · MultiDepthFusion
    simplefeature512x512   SimpleHead · Fusion RefineNet (feature)
    dpt512x512             DPTHead    · Fusion RefineNet (feature)
    dptmultidepth512x512   DPTHead    · MultiDepthFusion

Protocole (identique pour tous les modèles) :
  * Images val à la résolution native 1024×2048 (pas de resize, pas de TTA).
  * Inférence en fenêtre glissante 512×512, stride 341 (= protocole "slide" MMSeg
    pour les modèles entraînés en 512×512). `--eval-mode whole` reproduit à la place
    l'évaluation faite pendant l'entraînement (image entière en une passe).
  * Une matrice de confusion PAR IMAGE est conservée, ce qui permet :
      - mIoU / mAcc / aAcc / fwIoU, IoU par classe et par catégorie Cityscapes ;
      - intervalles de confiance à 95 % par bootstrap sur les images ;
      - comparaisons appariées entre modèles (mêmes rééchantillonnages), p-values
        bootstrap corrigées de Holm.
  * Visualisations qualitatives sur un sous-ensemble d'images FIXE (seed) et identique
    pour les 4 modèles.
  * Cartes d'attention du backbone : les couches 1..28 sont gelées donc identiques
    entre modèles (contrôle de cohérence), les couches 29..32 ont été fine-tunées
    différemment par chaque tête. Pour des points requêtes choisis automatiquement
    sur la GT (centre de l'objet le plus grand d'une classe), on calcule l'attention
    (moyenne des têtes) dans la fenêtre 512×512 que le modèle voit réellement en
    inférence. Métriques quantitatives par couche : distance moyenne d'attention et
    "lift sémantique" (masse d'attention sur les patches de même classe GT, divisée
    par la fraction de ces patches ; 1 = pas de préférence).

Toutes les sorties vont dans --output-dir (report.md, CSV, LaTeX, figures PNG).
Les résultats bruts par modèle sont mis en cache (cache/<modèle>.pt) : relancer avec
les mêmes paramètres ne ré-évalue pas, ce qui permet d'itérer sur les figures.

Usage :
  python src/test.py --data-root data/cityscapes --work-dirs work_dirs
  python src/test.py --data-root data/cityscapes --work-dirs work_dirs --max-images 20   # test rapide

  # Multi-GPU (Accelerate) : les images sont réparties entre GPU, résultats identiques
  accelerate launch --multi_gpu --num_processes 4 src/test.py --data-root data/cityscapes
"""

import argparse
import csv
import json
import os
import time
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from accelerate.utils import gather_object

from model import IJepaSegmentationModel
from dataset import CityscapesSegDataset, CITYSCAPES_MEAN, CITYSCAPES_STD

NUM_CLASSES = 19
IGNORE_INDEX = 255

log = print  # remplacé par accelerator.print dans main()

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

# Palette officielle Cityscapes (trainId → RGB)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# Catégories officielles Cityscapes (benchmark "category IoU")
CATEGORIES = {
    "flat": [0, 1],
    "construction": [2, 3, 4],
    "object": [5, 6, 7],
    "nature": [8, 9],
    "sky": [10],
    "human": [11, 12],
    "vehicle": [13, 14, 15, 16, 17, 18],
}

# Classes candidates pour les points requêtes d'attention, par ordre de priorité
# (objets fins / "things" d'abord, puis "stuff").
QUERY_CLASS_PRIORITY = [11, 13, 18, 12, 7, 5, 14, 15, 6, 1, 0, 8, 2, 10]


@dataclass(frozen=True)
class ModelSpec:
    name: str           # nom du dossier dans work_dirs/
    label: str          # nom affiché dans les figures
    decoder_type: str
    fusion_type: str
    color: str          # couleur catégorielle fixe (suit le modèle dans toutes les figures)
    note: str = ""      # écart de protocole d'entraînement éventuel


# Les 4 configurations ; le dossier d'un run = <préfixe><--run-suffix>
# (ex. simple512x512, dptmultidepth_fullres).
MODEL_CONFIGS = [
    ("simple", "Simple · MultiDepth", "simple", "multidepth", "#2a78d6"),
    ("simplefeature", "Simple · Feature", "simple", "feature", "#eb6834"),
    ("dpt", "DPT · Feature", "dpt", "feature", "#1baf7a"),
    ("dptmultidepth", "DPT · MultiDepth", "dpt", "multidepth", "#eda100"),
]
RUN_NOTES = {
    "simplefeature512x512": "entraîné avec batch_size=2 (4 pour les autres modèles)",
}


def make_model_zoo(suffix):
    return {
        prefix + suffix: ModelSpec(prefix + suffix, label, dec, fus, color, RUN_NOTES.get(prefix + suffix, ""))
        for prefix, label, dec, fus, color in MODEL_CONFIGS
    }


MODEL_ZOO = make_model_zoo("512x512")   # redéfini dans main() selon --run-suffix

# Couleurs neutres / rampes (séquentielle bleue, divergente bleu ↔ rouge avec milieu gris)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ERROR_RGB = np.array([227, 73, 72], dtype=np.float32)
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#b3261e", "#e34948", "#f4a9a8", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


# ===========================================================================
# Chargement des modèles
# ===========================================================================

def build_model(spec: ModelSpec, args, device):
    """
    Construit le modèle avec unfreeze_last_n=0 (encodeur en mode eval, pas de
    gradient checkpointing) puis charge la tête ET les blocs fine-tunés du backbone.
    Avec unfreeze_last_n>0, IJepaEncoder._sync_mode laisserait l'encodeur en mode
    train même après model.eval().
    """
    model = IJepaSegmentationModel(
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        layer_indices=tuple(args.layer_indices),
        unfreeze_last_n=0,
        decoder_type=spec.decoder_type,
        fusion_type=spec.fusion_type,
    )
    ckpt_path = os.path.join(args.work_dirs, spec.name, args.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "head" not in ckpt:
        raise KeyError(f"{ckpt_path} : clé 'head' absente (clés : {list(ckpt)})")

    model.head.load_state_dict(ckpt["head"])  # strict : échoue si la tête ne correspond pas
    backbone = ckpt.get("backbone", {})
    model.load_trainable_backbone_state_dict(backbone)

    tuned_layers = sorted({
        int(k.split("layers.")[1].split(".")[0]) + 1 for k in backbone if "layers." in k
    })
    if len(tuned_layers) != args.trained_unfreeze_last_n:
        log(f"  [!] {spec.name} : couches backbone chargées = {tuned_layers}, "
              f"attendu {args.trained_unfreeze_last_n} couches fine-tunées.")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    info = {
        "checkpoint": ckpt_path,
        "epoch": ckpt.get("epoch"),
        "train_val_miou": ckpt.get("miou"),
        "tuned_layers": tuned_layers,
        "head_params_M": sum(p.numel() for p in model.head.parameters()) / 1e6,
    }
    return model, info


# ===========================================================================
# Inférence
# ===========================================================================

def window_boxes(H, W, crop, stride):
    """Grille de fenêtres identique à MMSeg (EncoderDecoder.slide_inference)."""
    ch, cw = crop
    sh, sw = stride
    h_grids = max(H - ch + sh - 1, 0) // sh + 1
    w_grids = max(W - cw + sw - 1, 0) // sw + 1
    boxes = []
    for hi in range(h_grids):
        for wi in range(w_grids):
            y2, x2 = min(hi * sh + ch, H), min(wi * sw + cw, W)
            y1, x1 = max(y2 - ch, 0), max(x2 - cw, 0)
            boxes.append((y1, y2, x1, x2))
    return boxes


@torch.no_grad()
def predict_logits(model, image, args):
    """image : (1, 3, H, W) sur device → logits (C, H, W) float32."""
    amp = torch.autocast(device_type=image.device.type, dtype=torch.float16,
                         enabled=args.amp and image.device.type == "cuda")
    _, _, H, W = image.shape
    if args.eval_mode == "whole":
        with amp:
            return model(image)[0].float()

    boxes = window_boxes(H, W, tuple(args.crop_size), tuple(args.stride))
    logits = torch.zeros(NUM_CLASSES, H, W, device=image.device)
    count = torch.zeros(1, H, W, device=image.device)
    for i in range(0, len(boxes), args.window_batch):
        chunk = boxes[i:i + args.window_batch]
        crops = torch.cat([image[:, :, y1:y2, x1:x2] for y1, y2, x1, x2 in chunk])
        with amp:
            out = model(crops).float()
        for (y1, y2, x1, x2), o in zip(chunk, out):
            logits[:, y1:y2, x1:x2] += o
            count[:, y1:y2, x1:x2] += 1
    return logits / count


def image_confusion(pred, gt):
    """Matrice de confusion (GT en lignes, prédiction en colonnes), sur device."""
    valid = gt != IGNORE_INDEX
    idx = gt[valid] * NUM_CLASSES + pred[valid]
    return torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


# ===========================================================================
# Attention du backbone
# ===========================================================================

class AttentionProbe:
    """
    Capture Q et K de chaque bloc transformer via des hooks sur les projections
    q/k, puis recalcule softmax(QKᵀ/√d). Évite output_attentions=True (qui force
    l'implémentation eager et renverrait les 32 matrices d'un coup).
    I-JEPA n'a pas de token CLS : tous les tokens sont des patches.
    """

    def __init__(self, model):
        hf = model.encoder.encoder
        self.num_heads = hf.config.num_attention_heads
        self.q, self.k, self._hooks = {}, {}, []
        for i, layer in enumerate(model.encoder._get_transformer_layers()):
            att = layer.attention
            # Nommage HF récent (q_proj/k_proj) ou ancien style ViT (attention.query/key)
            if hasattr(att, "q_proj"):
                q_mod, k_mod = att.q_proj, att.k_proj
            else:
                q_mod, k_mod = att.attention.query, att.attention.key
            self._hooks.append(q_mod.register_forward_hook(self._store(self.q, i + 1)))
            self._hooks.append(k_mod.register_forward_hook(self._store(self.k, i + 1)))
        self.num_layers = len(self._hooks) // 2

    @staticmethod
    def _store(buffer, layer_idx):
        def hook(module, inputs, output):
            buffer[layer_idx] = output.detach()
        return hook

    def attention_rows(self, layer_idx, rows):
        """
        Lignes `rows` (slice) de la matrice d'attention → (heads, R, N) float32, pour le
        premier élément du batch. Calcul par blocs de lignes : en pleine résolution
        (N = 73×146 tokens), la matrice complète ferait ~7 Go par couche.
        """
        q, k = self.q[layer_idx][0].float(), self.k[layer_idx][0].float()
        N, D = q.shape
        d = D // self.num_heads
        q = q[rows].reshape(-1, self.num_heads, d).transpose(0, 1)
        k = k.view(N, self.num_heads, d).transpose(0, 1)
        return torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, dim=-1)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self.q.clear()
        self.k.clear()


def select_queries(gt, max_queries, min_area):
    """
    Points requêtes déterministes à partir de la GT : pour chaque classe (ordre de
    QUERY_CLASS_PRIORITY), le pixel le plus intérieur (max de la transformée de
    distance) de la plus grande composante connexe.
    """
    queries = []
    for c in QUERY_CLASS_PRIORITY:
        mask = (gt == c).astype(np.uint8)
        if mask.sum() < min_area:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[biggest, cv2.CC_STAT_AREA] < min_area:
            continue
        # Padding à 0 : sinon distanceTransform considère l'extérieur de l'image comme
        # "loin de tout bord" et place la requête des grandes classes sur le bord.
        comp = np.pad((lab == biggest).astype(np.uint8), 1)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        queries.append((c, int(y), int(x)))
        if len(queries) == max_queries:
            break
    return queries


def patch_labels(gt_crop, patch, grid_h, grid_w):
    """Label majoritaire de chaque patch (-1 si uniquement 'ignore')."""
    g = gt_crop[:grid_h * patch, :grid_w * patch].reshape(grid_h, patch, grid_w, patch)
    counts = np.stack([(g == c).sum(axis=(1, 3)) for c in range(NUM_CLASSES)], axis=-1)
    lab = counts.argmax(-1)
    lab[counts.sum(-1) == 0] = -1
    return lab


def build_attention_windows(dataset, indices, args, patch):
    """Fenêtres --attn-crop-size centrées (avec clamp) sur chaque point requête."""
    ch, cw = args.attn_crop_size
    grid_h, grid_w = ch // patch, cw // patch
    windows = []
    for img_idx in indices:
        image, label = dataset[img_idx]
        gt = label.numpy()
        H, W = gt.shape
        for q_idx, (c, y, x) in enumerate(select_queries(gt, args.attn_queries, args.attn_min_area)):
            y1 = int(np.clip(y - ch // 2, 0, H - ch))
            x1 = int(np.clip(x - cw // 2, 0, W - cw))
            gt_crop = gt[y1:y1 + ch, x1:x1 + cw]
            labels = patch_labels(gt_crop, patch, grid_h, grid_w)
            windows.append({
                "img_idx": img_idx, "q_idx": q_idx, "cls": c,
                "box": (y1, x1), "query_px": (y - y1, x - x1),
                "query_tok": (min((y - y1) // patch, grid_h - 1), min((x - x1) // patch, grid_w - 1)),
                "labels": labels,
                "image": image[:, y1:y1 + ch, x1:x1 + cw].clone(),
            })
    return windows, (grid_h, grid_w)


@torch.no_grad()
def analyse_attention(model, windows, grid, args, device):
    """
    Pour chaque fenêtre et chaque couche :
      * distance moyenne d'attention (px), par tête, sur toutes les requêtes ;
      * lift sémantique moyen, par tête (requêtes avec label valide) ;
      * pour le point requête : carte d'attention (moyenne des têtes) aux couches
        --attn-layers, masse sur la classe du point, lift et distance.
    """
    probe = AttentionProbe(model)
    L, heads = probe.num_layers, probe.num_heads
    gh, gw = grid
    patch = model.encoder.patch_size
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    coords = torch.stack([ys.flatten(), xs.flatten()], -1).float().to(device) * patch
    dist = torch.cdist(coords, coords)                                  # (N, N) en px

    md_sum = torch.zeros(L, heads, device=device)
    lift_sum = torch.zeros(L, heads, device=device)
    maps, query_rows = {}, []
    amp = torch.autocast(device_type=device.type, dtype=torch.float16,
                         enabled=args.amp and device.type == "cuda")

    for w in windows:
        with amp:
            model.encoder(w["image"].unsqueeze(0).to(device))
        lab = torch.from_numpy(w["labels"].flatten()).to(device)
        valid = lab >= 0
        same = (lab[:, None] == lab[None, :]) & valid[None, :]           # (N, N)
        frac = same.float().mean(-1)                                     # (N,)
        q_tok = w["query_tok"][0] * gw + w["query_tok"][1]
        # Objets fins (poteaux…) : la classe du point peut n'être majoritaire dans
        # aucun patch → masse/lift non définis pour cette requête.
        q_mask = lab == w["cls"]
        q_frac = q_mask.float().mean().item()

        N = gh * gw
        ok = valid & (frac > 0)
        n_ok = ok.sum().clamp(min=1)
        for layer in range(1, L + 1):
            md = torch.zeros(heads, device=device)
            lift = torch.zeros(heads, device=device)
            for start in range(0, N, args.attn_chunk):
                rows = slice(start, min(start + args.attn_chunk, N))
                A = probe.attention_rows(layer, rows)                    # (h, R, N)
                md += (A * dist[rows]).sum(-1).sum(-1)
                mass = (A * same[rows]).sum(-1)                          # (h, R)
                ok_r = ok[rows]
                lift += (mass[:, ok_r] / frac[rows][ok_r]).sum(-1)
            md_sum[layer - 1] += md / N
            lift_sum[layer - 1] += lift / n_ok

            a = probe.attention_rows(layer, slice(q_tok, q_tok + 1))[:, 0].mean(0)   # (N,)
            q_mass = a[q_mask].sum().item() if q_frac > 0 else float("nan")
            query_rows.append({
                "img_idx": w["img_idx"], "q_idx": w["q_idx"], "cls": CLASS_NAMES[w["cls"]],
                "layer": layer, "same_class_mass": q_mass,
                "same_class_frac": q_frac, "lift": q_mass / q_frac if q_frac > 0 else float("nan"),
                "attn_distance_px": (a * dist[q_tok]).sum().item(),
            })
            if layer in args.attn_layers:
                maps[(w["img_idx"], w["q_idx"], layer)] = a.view(gh, gw).cpu().numpy()

    probe.remove()
    n = max(len(windows), 1)
    return {
        "mean_distance": (md_sum / n).cpu().numpy(),     # (L, heads)
        "semantic_lift": (lift_sum / n).cpu().numpy(),   # (L, heads)
        "maps": maps,
        "query_rows": query_rows,
    }


# ===========================================================================
# Évaluation d'un modèle (avec cache)
# ===========================================================================

def cache_signature(spec, args, n_images, vis_indices, attn_windows):
    ckpt_path = os.path.join(args.work_dirs, spec.name, args.ckpt_name)
    return {
        "model": spec.name, "ckpt": os.path.abspath(ckpt_path),
        "ckpt_mtime": os.path.getmtime(ckpt_path),
        "eval_mode": args.eval_mode, "crop": list(args.crop_size), "stride": list(args.stride),
        "amp": args.amp, "n_images": n_images, "vis": list(vis_indices),
        "attn_crop": list(args.attn_crop_size),
        "attn_windows": [(w["img_idx"], w["cls"], w["box"], w["query_px"]) for w in attn_windows],
        "attn_layers": list(args.attn_layers),
        "layer_indices": list(args.layer_indices),
    }


def evaluate_model(spec, args, dataset, n_images, vis_indices, attn_windows, grid, accelerator):
    """
    Chaque processus évalue les images rank, rank+world, ... (partition exacte, aucune
    image dupliquée) ; les résultats par image sont rassemblés sur le processus
    principal, qui fait seul l'analyse d'attention. Retourne None hors processus principal.
    """
    device = accelerator.device
    rank, world = accelerator.process_index, accelerator.num_processes
    # main_process_first : un seul téléchargement HF si le cache est vide
    with accelerator.main_process_first():
        model, info = build_model(spec, args, device)
    log(f"  checkpoint : epoch {info['epoch']}, mIoU val (entraînement) "
          f"{info['train_val_miou']}, couches fine-tunées {info['tuned_layers']}")

    shard = list(range(rank, n_images, world))
    loader = DataLoader(Subset(dataset, shard), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    local = []   # (idx, confusion, prédiction si image de visualisation, temps)
    vis_set = set(vis_indices)

    for k, (image, label) in enumerate(loader):
        i = shard[k]
        image = image.to(device, non_blocking=True)
        label = label[0].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = predict_logits(model, image, args).argmax(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        cm = image_confusion(pred, label).cpu().numpy()
        local.append((i, cm, pred.cpu().numpy().astype(np.uint8) if i in vis_set else None, elapsed))
        if (k + 1) % 50 == 0 or k + 1 == len(shard):
            running = metrics_from_cm(sum(c for _, c, _, _ in local))["miou"]
            log(f"  [{(k + 1) * world}/{n_images}] mIoU cumulé (shard principal) {100 * running:.2f}")

    gathered = gather_object(local)
    attn = None
    if accelerator.is_main_process and attn_windows:
        attn = analyse_attention(model, attn_windows, grid, args, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None

    per_image_cm = np.zeros((n_images, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    vis_preds = {}
    for i, cm, pred, _ in gathered:
        per_image_cm[i] = cm
        if pred is not None:
            vis_preds[i] = pred
    assert len({g[0] for g in gathered}) == n_images, "images manquantes après gather"
    # Latence mesurée sur un GPU (processus principal), hors 5 premières images (warm-up)
    times = [t for _, _, _, t in local]
    warm = times[5:] if len(times) > 10 else times
    return {
        "info": info,
        "per_image_cm": per_image_cm,
        "vis_preds": vis_preds,
        "latency_ms": 1000 * float(np.mean(warm)),
        "attn": attn,
    }


# ===========================================================================
# Métriques & statistiques
# ===========================================================================

def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt, pr = cm.sum(1), cm.sum(0)
    union = gt + pr - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        acc = np.where(gt > 0, tp / gt, np.nan)
    return {
        "iou": iou, "acc": acc,
        "miou": float(np.nanmean(iou)), "macc": float(np.nanmean(acc)),
        "aacc": float(tp.sum() / max(cm.sum(), 1)),
        "fwiou": float(np.nansum(gt / max(gt.sum(), 1) * np.nan_to_num(iou))),
    }


def category_cm(cm):
    M = np.zeros((len(CATEGORIES), NUM_CLASSES))
    for j, classes in enumerate(CATEGORIES.values()):
        M[j, classes] = 1
    return M @ cm @ M.T


def bootstrap_iou(per_image_cm, resample_idx):
    """
    Rééchantillonnage des images (avec remise). Le mIoU de chaque réplique est
    calculé sur la matrice de confusion cumulée (comme le mIoU "dataset-level"),
    pas en moyennant des mIoU par image. → (B, C) IoU, (B,) mIoU.
    """
    B, N = resample_idx.shape
    weights = np.stack([np.bincount(r, minlength=N) for r in resample_idx]).astype(np.float64)
    cms = (weights @ per_image_cm.reshape(N, -1).astype(np.float64)).reshape(B, NUM_CLASSES, NUM_CLASSES)
    tp = np.diagonal(cms, axis1=1, axis2=2)
    union = cms.sum(2) + cms.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    return iou, np.nanmean(iou, axis=1)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def compute_statistics(results, names, args):
    n_images = next(iter(results.values()))["per_image_cm"].shape[0]
    rng = np.random.default_rng(args.seed)
    resample_idx = rng.integers(0, n_images, size=(args.bootstrap, n_images))

    stats = {}
    for name in names:
        cm = results[name]["per_image_cm"]
        total = cm.sum(0)
        m = metrics_from_cm(total)
        boot_iou, boot_miou = bootstrap_iou(cm, resample_idx)
        stats[name] = {
            **m,
            "cat_iou": metrics_from_cm(category_cm(total))["iou"],
            "cm": total,
            "boot_miou": boot_miou,
            "boot_iou": boot_iou,
            "miou_ci": np.percentile(boot_miou, [2.5, 97.5]),
            "iou_ci": np.nanpercentile(boot_iou, [2.5, 97.5], axis=0),
            "per_image_miou": np.array([metrics_from_cm(c)["miou"] for c in cm]),
        }
        stats[name]["cat_miou"] = float(np.nanmean(stats[name]["cat_iou"]))

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            delta = stats[b]["boot_miou"] - stats[a]["boot_miou"]
            p = min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
            pairs.append({
                "a": a, "b": b,
                "delta": stats[b]["miou"] - stats[a]["miou"],
                "ci": np.percentile(delta, [2.5, 97.5]), "p": p,
            })
    if pairs:
        for pair, p_adj in zip(pairs, holm(np.array([p["p"] for p in pairs]))):
            pair["p_holm"] = p_adj
    return stats, pairs


# ===========================================================================
# Figures
# ===========================================================================

def denormalize(image_t):
    mean = torch.tensor(CITYSCAPES_MEAN).view(3, 1, 1)
    std = torch.tensor(CITYSCAPES_STD).view(3, 1, 1)
    return (image_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def colorize(label):
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    valid = label < NUM_CLASSES
    rgb[valid] = CITYSCAPES_PALETTE[label[valid]]
    return rgb


def error_map(image_rgb, pred, gt):
    """Pixels corrects : image en gris atténué ; erreurs : rouge ; ignore : noir."""
    gray = image_rgb.mean(-1, keepdims=True).repeat(3, -1) * 255 * 0.55
    out = gray.astype(np.float32)
    wrong = (pred != gt) & (gt != IGNORE_INDEX)
    out[wrong] = ERROR_RGB
    out[gt == IGNORE_INDEX] = 0
    return out.astype(np.uint8)


def class_legend_handles():
    return [Patch(facecolor=CITYSCAPES_PALETTE[i] / 255, edgecolor="none", label=n)
            for i, n in enumerate(CLASS_NAMES)]


def fig_miou_ci(stats, specs, path):
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(specs) + 1.0))
    for y, s in enumerate(specs):
        st = stats[s.name]
        lo, hi = 100 * st["miou_ci"]
        ax.plot([lo, hi], [y, y], color=s.color, lw=2, solid_capstyle="round")
        ax.plot(100 * st["miou"], y, "o", ms=9, color=s.color, mec="white", mew=2)
        ax.text(hi + 0.15, y, f"{100 * st['miou']:.2f}  [{lo:.2f}, {hi:.2f}]",
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(specs)), [s.label for s in specs])
    ax.invert_yaxis()
    ax.set_xlabel("mIoU (%) — IC 95 % bootstrap sur les images")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    xmin = min(100 * stats[s.name]["miou_ci"][0] for s in specs)
    xmax = max(100 * stats[s.name]["miou_ci"][1] for s in specs)
    ax.set_xlim(xmin - 0.5, xmax + 3.0)
    ax.set_title("mIoU Cityscapes val", loc="left", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def fig_per_class_iou(stats, specs, path):
    n = len(specs)
    width = 0.8 / n
    x = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(16, 4.8))
    for k, s in enumerate(specs):
        st = stats[s.name]
        iou = 100 * st["iou"]
        err = np.abs(100 * st["iou_ci"] - iou[None])
        ax.bar(x + (k - (n - 1) / 2) * width, iou, width, color=s.color, label=s.label,
               edgecolor="white", linewidth=1.0,
               yerr=err, error_kw={"elinewidth": 0.8, "ecolor": INK_2, "capsize": 0})
    ax.set_xticks(x, CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=n, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    ax.set_title("IoU par classe (barres : IC 95 % bootstrap)", loc="left", fontsize=11, pad=24)
    fig.savefig(path)
    plt.close(fig)


def fig_delta_heatmap(stats, specs, baseline, path):
    others = [s for s in specs if s.name != baseline.name]
    if not others:
        return
    base = stats[baseline.name]
    cols = CLASS_NAMES + ["mIoU"]
    D = np.zeros((len(others), len(cols)))
    sig = np.zeros_like(D, dtype=bool)
    for r, s in enumerate(others):
        st = stats[s.name]
        d_boot = np.concatenate([st["boot_iou"] - base["boot_iou"],
                                 (st["boot_miou"] - base["boot_miou"])[:, None]], axis=1)
        D[r] = 100 * np.append(st["iou"] - base["iou"], st["miou"] - base["miou"])
        lo, hi = np.nanpercentile(d_boot, [2.5, 97.5], axis=0)
        sig[r] = (lo > 0) | (hi < 0)
    vlim = max(np.nan_to_num(np.abs(D)).max(), 0.5)
    fig, ax = plt.subplots(figsize=(16, 0.6 * len(others) + 1.6))
    im = ax.imshow(D, cmap=DIVERGING, vmin=-vlim, vmax=vlim, aspect="auto")
    for r in range(D.shape[0]):
        for c in range(D.shape[1]):
            txt = "—" if np.isnan(D[r, c]) else f"{D[r, c]:+.1f}" + ("*" if sig[r, c] else "")
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if abs(D[r, c]) > 0.6 * vlim else INK,
                    fontweight="bold" if sig[r, c] else "normal")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(others)), [s.label for s in others])
    ax.axvline(len(cols) - 1.5, color="white", lw=3)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Δ IoU (points)")
    ax.set_title(f"Δ IoU par rapport à {baseline.label}   (* : IC 95 % bootstrap excluant 0)",
                 loc="left", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def fig_confusions(stats, specs, path):
    n = len(specs)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 6.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, s in zip(axes.flat, specs):
        cm = stats[s.name]["cm"].astype(np.float64)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=7)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vérité terrain")
        ax.set_title(f"{s.label} — mIoU {100 * stats[s.name]['miou']:.2f}", loc="left")
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Fraction des pixels GT (rappel)")
    fig.suptitle("Matrices de confusion normalisées par ligne", x=0.05, ha="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def fig_qualitative(idx, image_rgb, gt, preds, specs, stats, path):
    n = len(specs)
    fig, axes = plt.subplots(2, n + 2, figsize=(3.6 * (n + 2), 4.4))
    for ax in axes.flat:
        ax.axis("off")

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title(f"Image val #{idx}")
    axes[0, 1].imshow(colorize(gt))
    axes[0, 1].set_title("Vérité terrain")
    for k, s in enumerate(specs):
        axes[0, k + 2].imshow(colorize(preds[s.name]))
        axes[0, k + 2].set_title(f"{s.label}\nmIoU image {100 * stats[s.name]['per_image_miou'][idx]:.1f}",
                                 color=INK)
        axes[1, k + 2].imshow(error_map(image_rgb, preds[s.name], gt))
        wrong = ((preds[s.name] != gt) & (gt != IGNORE_INDEX)).sum() / max((gt != IGNORE_INDEX).sum(), 1)
        axes[1, k + 2].set_title(f"Erreurs : {100 * wrong:.1f} % des pixels")

    n_distinct = distinct_count(np.stack([preds[s.name] for s in specs]))
    im = axes[1, 0].imshow(n_distinct, cmap=ListedColormap(["#f0efec", "#86b6ef", "#256abf", "#0d366b"][:n]),
                           vmin=0.5, vmax=n + 0.5)
    axes[1, 0].set_title("Désaccord : nb de classes\nprédites différentes")
    cb = fig.colorbar(im, ax=axes[1, 0], fraction=0.035, pad=0.02, ticks=range(1, n + 1))
    cb.ax.tick_params(labelsize=7)
    axes[1, 1].legend(handles=class_legend_handles(), loc="center", ncol=2, fontsize=6.5,
                      frameon=False, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def distinct_count(stack):
    """Nombre de labels distincts prédits par pixel parmi les modèles (stack : (M, H, W))."""
    count = np.ones(stack.shape[1:], dtype=np.int64)
    for m in range(1, stack.shape[0]):
        new = np.ones(stack.shape[1:], dtype=bool)
        for prev in range(m):
            new &= stack[m] != stack[prev]
        count += new
    return count


def fig_qualitative_overview(vis, images, gts, results, specs, path, max_rows=8):
    vis = vis[:max_rows]
    n = len(specs)
    fig, axes = plt.subplots(len(vis), n + 2, figsize=(2.9 * (n + 2), 1.55 * len(vis) + 0.5),
                             squeeze=False)
    headers = ["Image", "Vérité terrain"] + [s.label for s in specs]
    for r, idx in enumerate(vis):
        panels = [images[idx], colorize(gts[idx])] + \
                 [colorize(results[s.name]["vis_preds"][idx]) for s in specs]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p[::4, ::4])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if r == 0:
                ax.set_title(headers[c], fontsize=9)
        axes[r, 0].set_ylabel(f"#{idx}", fontsize=8)
    fig.subplots_adjust(wspace=0.02, hspace=0.04)
    fig.savefig(path)
    plt.close(fig)


def _crop_rgb(window, patch, grid):
    gh, gw = grid
    return denormalize(window["image"])[:gh * patch, :gw * patch]


def _overlay(ax, rgb, heat, vmax, patch, query_px, title=None):
    h, w = rgb.shape[:2]
    ax.imshow(rgb)
    im = ax.imshow(heat, cmap="inferno", alpha=0.6, vmin=0, vmax=vmax,
                   extent=(0, w, h, 0), interpolation="bilinear")
    ax.plot(min(query_px[1], w - 1), min(query_px[0], h - 1), marker="+", ms=14, mew=2.5, color="#00e5ff")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def _grid_with_colorbars(rows, cols, size=2.9, aspect=1.0):
    """Grille de panneaux (largeur/hauteur = aspect) + une colonne de colorbars (une par ligne)."""
    fig = plt.figure(figsize=(size * aspect * cols + 0.6, size * rows + 0.4))
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.05], wspace=0.08, hspace=0.15)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)])
    caxes = [fig.add_subplot(gs[r, cols]) for r in range(rows)]
    return fig, axes, caxes


def fig_attention(windows, results, specs, args, grid, patch, out_dir, ref_layer, final_layer):
    """
    Deux figures par image :
      attn_img<idx>.png          lignes = points requêtes ; colonnes = crop + GT,
                                 couche gelée de référence (partagée), puis chaque
                                 modèle à la dernière couche.
      attn_img<idx>_q<j>.png     lignes = couches de --attn-layers ; colonnes = modèles.
    Échelle : attention × N (1 = attention uniforme), même vmax sur une ligne pour
    que les modèles soient comparables.
    """
    N = grid[0] * grid[1]
    by_img = {}
    for w in windows:
        by_img.setdefault(w["img_idx"], []).append(w)
    fig_imgs = list(by_img)[:args.num_attn_figures]

    aspect = grid[1] / grid[0]
    has_ref = ref_layer >= 1        # False si tout le backbone a été fine-tuné
    off = 2 if has_ref else 1       # colonnes avant celles des modèles
    for img_idx in fig_imgs:
        ws = by_img[img_idx]
        n_cols = len(specs) + off
        fig, axes, caxes = _grid_with_colorbars(len(ws), n_cols, aspect=aspect)
        for r, w in enumerate(ws):
            rgb = _crop_rgb(w, patch, grid)
            key_ref = (img_idx, w["q_idx"], ref_layer)
            key_fin = (img_idx, w["q_idx"], final_layer)
            maps = ([results[specs[0].name]["attn"]["maps"][key_ref] * N] if has_ref else []) + \
                   [results[s.name]["attn"]["maps"][key_fin] * N for s in specs]
            vmax = max(np.percentile(m, 99.5) for m in maps)

            ax = axes[r, 0]
            gt_crop = colorize(np.where(w["labels"] < 0, IGNORE_INDEX, w["labels"]))
            ax.imshow(rgb)
            ax.imshow(cv2.resize(gt_crop, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST), alpha=0.35)
            ax.plot(w["query_px"][1], w["query_px"][0], marker="+", ms=14, mew=2.5, color="#00e5ff")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            ax.set_ylabel(f"requête : {CLASS_NAMES[w['cls']]}", fontsize=9)
            if r == 0:
                ax.set_title(f"Crop {args.attn_crop_size[0]}×{args.attn_crop_size[1]} + GT par patch",
                             fontsize=9)

            if has_ref:
                _overlay(axes[r, 1], rgb, maps[0], vmax, patch, w["query_px"],
                         f"Couche {ref_layer} (gelée,\nidentique pour tous)" if r == 0 else None)
            for k, s in enumerate(specs):
                im = _overlay(axes[r, k + off], rgb, maps[k + off - 1], vmax, patch, w["query_px"],
                              f"{s.label}\ncouche {final_layer}" if r == 0 else None)
            fig.colorbar(im, cax=caxes[r]).set_label("attention × N", fontsize=8)
        fig.suptitle(f"Attention du point requête (moyenne des têtes) — image val #{img_idx}",
                     x=0.02, ha="left", fontsize=11)
        fig.savefig(os.path.join(out_dir, f"attn_img{img_idx:03d}.png"))
        plt.close(fig)

        for w in ws:
            rgb = _crop_rgb(w, patch, grid)
            layers = sorted(args.attn_layers)
            fig, axes, caxes = _grid_with_colorbars(len(layers), len(specs), aspect=aspect)
            for r, layer in enumerate(layers):
                maps = [results[s.name]["attn"]["maps"][(img_idx, w["q_idx"], layer)] * N for s in specs]
                vmax = max(np.percentile(m, 99.5) for m in maps)
                status = ("gelée" if layer <= ref_layer else
                          "non lue par la tête" if layer > final_layer else "fine-tunée")
                for k, s in enumerate(specs):
                    im = _overlay(axes[r, k], rgb, maps[k], vmax, patch, w["query_px"],
                                  s.label if r == 0 else None)
                axes[r, 0].set_ylabel(f"couche {layer} ({status})", fontsize=9)
                fig.colorbar(im, cax=caxes[r]).set_label("attention × N", fontsize=8)
            fig.suptitle(f"Image #{img_idx} — requête « {CLASS_NAMES[w['cls']]} » : attention par couche",
                         x=0.02, ha="left", fontsize=11)
            fig.savefig(os.path.join(out_dir, f"attn_img{img_idx:03d}_q{w['q_idx']}_layers.png"))
            plt.close(fig)


def fig_attention_layer_stats(results, specs, ref_layer, final_layer, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    panels = [("mean_distance", "Distance moyenne d'attention (px)"),
              ("semantic_lift", "Lift sémantique (1 = aucune préférence)")]
    for ax, (key, ylabel) in zip(axes, panels):
        for s in specs:
            vals = results[s.name]["attn"][key].mean(1)
            layers = np.arange(1, len(vals) + 1)
            ax.plot(layers, vals, color=s.color, lw=2, marker="o", ms=4, label=s.label)
        if ref_layer >= 1:
            ax.axvspan(0.5, ref_layer + 0.5, color="#f0efec", zorder=0)
            ax.text(ref_layer / 2, ax.get_ylim()[1], "couches gelées (identiques)", ha="center",
                    va="top", fontsize=8, color=INK_2)
        if final_layer < len(vals):
            ax.axvspan(final_layer + 0.5, len(vals) + 0.5, color="#f0efec", zorder=0)
        ax.set_xlim(0.5, len(vals) + 0.5)
        ax.set_xlabel("Couche transformer")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Statistiques d'attention par couche (moyenne des têtes et des fenêtres)",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ===========================================================================
# Exports texte
# ===========================================================================

def write_csvs(stats, specs, pairs, results, out_dir):
    with open(os.path.join(out_dir, "per_class_iou.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "class", "iou", "iou_ci_low", "iou_ci_high", "acc"])
        for s in specs:
            st = stats[s.name]
            for c, name in enumerate(CLASS_NAMES):
                wr.writerow([s.name, name, st["iou"][c], st["iou_ci"][0, c], st["iou_ci"][1, c], st["acc"][c]])

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "miou", "miou_ci_low", "miou_ci_high", "macc", "aacc", "fwiou",
                     "category_miou", "latency_ms", "head_params_M", "epoch"])
        for s in specs:
            st, r = stats[s.name], results[s.name]
            wr.writerow([s.name, st["miou"], *st["miou_ci"], st["macc"], st["aacc"], st["fwiou"],
                         st["cat_miou"], r["latency_ms"], r["info"]["head_params_M"], r["info"]["epoch"]])

    with open(os.path.join(out_dir, "pairwise_tests.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model_a", "model_b", "delta_miou_b_minus_a", "ci_low", "ci_high", "p_bootstrap", "p_holm"])
        for p in pairs:
            wr.writerow([p["a"], p["b"], p["delta"], *p["ci"], p["p"], p["p_holm"]])

    with open(os.path.join(out_dir, "per_image_miou.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_idx"] + [s.name for s in specs])
        for i in range(len(stats[specs[0].name]["per_image_miou"])):
            wr.writerow([i] + [stats[s.name]["per_image_miou"][i] for s in specs])

    rows = [r | {"model": s.name} for s in specs if results[s.name]["attn"]
            for r in results[s.name]["attn"]["query_rows"]]
    if rows:
        with open(os.path.join(out_dir, "attention_queries.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


def write_latex(stats, specs, path):
    best = {c: max(stats[s.name]["iou"][c] for s in specs) for c in range(NUM_CLASSES)}
    best_miou = max(stats[s.name]["miou"] for s in specs)
    short = [n[:5] + "." if len(n) > 6 else n for n in CLASS_NAMES]
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{l" + "c" * (NUM_CLASSES + 1) + "}",
        "\\toprule",
        "Modèle & " + " & ".join(short) + " & mIoU \\\\",
        "\\midrule",
    ]
    for s in specs:
        st = stats[s.name]
        cells = []
        for c in range(NUM_CLASSES):
            v = f"{100 * st['iou'][c]:.1f}"
            cells.append(f"\\textbf{{{v}}}" if st["iou"][c] == best[c] else v)
        m = f"{100 * st['miou']:.1f}"
        cells.append(f"\\textbf{{{m}}}" if st["miou"] == best_miou else m)
        lines.append(f"{s.label.replace('·', '/')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{IoU par classe (\\%) sur Cityscapes val, inférence fenêtre glissante 512$\\times$512.}",
              "\\end{table*}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def top_confusions(cm, k=5):
    cm = cm.astype(np.float64)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    np.fill_diagonal(norm, 0)
    flat = np.argsort(norm.flatten())[::-1][:k]
    return [(CLASS_NAMES[i // NUM_CLASSES], CLASS_NAMES[i % NUM_CLASSES], norm.flat[i]) for i in flat]


def write_report(stats, specs, pairs, results, args, n_images, vis, attn_windows, ref_layer,
                 final_layer, frozen_check, path):
    pct = lambda v: f"{100 * v:.2f}"
    L = []
    L.append("# Comparaison des têtes de segmentation I-JEPA — Cityscapes val\n")
    L.append("## Protocole\n")
    mode = (f"fenêtre glissante {args.crop_size[0]}×{args.crop_size[1]}, stride "
            f"{args.stride[0]}×{args.stride[1]}" if args.eval_mode == "slide"
            else "image entière 1024×2048 en une passe")
    L.append(f"- Images : {n_images} images val, résolution native 1024×2048, sans TTA.")
    L.append(f"- Inférence : {mode}, {'fp16 (autocast)' if args.amp else 'fp32'}.")
    L.append(f"- Checkpoint : `{args.ckpt_name}` de chaque run.")
    L.append(f"- IC 95 % : bootstrap percentile sur les images ({args.bootstrap} répliques, "
             f"seed {args.seed}), mêmes répliques pour tous les modèles (tests appariés).")
    L.append("")
    L.append("| Modèle | Décodeur | Fusion | Epoch | mIoU val (train) | Couches fine-tunées | Params tête (M) | Note |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in specs:
        info = results[s.name]["info"]
        tv = f"{100 * info['train_val_miou']:.2f}" if info["train_val_miou"] is not None else "—"
        L.append(f"| {s.label} | {s.decoder_type} | {s.fusion_type} | {info['epoch']} | {tv} | "
                 f"{info['tuned_layers']} | {info['head_params_M']:.2f} | {s.note} |")

    L.append("\n## Résultats globaux\n")
    L.append("| Modèle | mIoU [IC 95 %] | mAcc | aAcc | fwIoU | mIoU catégories | Latence (ms/img) |")
    L.append("|---|---|---|---|---|---|---|")
    best = max(stats[s.name]["miou"] for s in specs)
    for s in specs:
        st = stats[s.name]
        m = f"{pct(st['miou'])} [{pct(st['miou_ci'][0])}, {pct(st['miou_ci'][1])}]"
        if st["miou"] == best:
            m = f"**{m}**"
        L.append(f"| {s.label} | {m} | {pct(st['macc'])} | {pct(st['aacc'])} | {pct(st['fwiou'])} | "
                 f"{pct(st['cat_miou'])} | {results[s.name]['latency_ms']:.0f} |")
    L.append("\n![mIoU](miou_ci.png)\n")

    if pairs:
        L.append("### Comparaisons appariées (Δ mIoU = B − A)\n")
        L.append("| A | B | Δ mIoU (pts) | IC 95 % | p bootstrap | p Holm |")
        L.append("|---|---|---|---|---|---|")
        for p in pairs:
            la, lb = MODEL_ZOO[p["a"]].label, MODEL_ZOO[p["b"]].label
            flag = " ✱" if p["p_holm"] < 0.05 else ""
            L.append(f"| {la} | {lb} | {100 * p['delta']:+.2f} | [{100 * p['ci'][0]:+.2f}, "
                     f"{100 * p['ci'][1]:+.2f}] | {p['p']:.3f} | {p['p_holm']:.3f}{flag} |")
        L.append("\n✱ : significatif à 5 % après correction de Holm.\n")

    L.append("## IoU par classe\n")
    L.append("| Classe | " + " | ".join(s.label for s in specs) + " |")
    L.append("|---|" + "---|" * len(specs))
    for c, name in enumerate(CLASS_NAMES):
        vals = [stats[s.name]["iou"][c] for s in specs]
        top = np.nanmax(vals)
        cells = [f"**{pct(v)}**" if v == top else pct(v) for v in vals]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("\n![IoU par classe](per_class_iou.png)\n")
    L.append(f"![Δ IoU](per_class_delta.png)\n")

    L.append("### IoU par catégorie Cityscapes\n")
    L.append("| Catégorie | " + " | ".join(s.label for s in specs) + " |")
    L.append("|---|" + "---|" * len(specs))
    for j, cat in enumerate(CATEGORIES):
        L.append(f"| {cat} | " + " | ".join(pct(stats[s.name]["cat_iou"][j]) for s in specs) + " |")

    L.append("\n### Confusions principales (fraction des pixels GT de la classe)\n")
    for s in specs:
        conf = ", ".join(f"{a} → {b} ({100 * v:.1f} %)" for a, b, v in top_confusions(stats[s.name]["cm"]))
        L.append(f"- **{s.label}** : {conf}")
    L.append("\n![Confusions](confusion_matrices.png)\n")

    L.append("## Qualitatif\n")
    L.append(f"Images (tirées avec seed {args.seed}, identiques pour tous les modèles) : {vis}.\n")
    L.append("![Vue d'ensemble](qualitative/overview.png)\n")
    L.append("Détails par image : `qualitative/img<idx>.png` (prédictions, cartes d'erreurs, "
             "carte de désaccord entre modèles) ; prédictions pleine résolution dans `predictions/`.\n")

    if attn_windows and all(results[s.name]["attn"] for s in specs):
        n_img = len({w["img_idx"] for w in attn_windows})
        L.append("## Attention du backbone\n")
        frozen_txt = (f"Couches 1–{ref_layer} gelées (identiques entre modèles), " if ref_layer >= 1
                      else "Backbone entièrement fine-tuné (aucune couche de référence commune), ")
        L.append(f"{len(attn_windows)} fenêtres {args.attn_crop_size[0]}×{args.attn_crop_size[1]} centrées sur "
                 f"des points requêtes choisis sur la GT ({n_img} images). {frozen_txt}"
                 f"couches {ref_layer + 1}–{final_layer} fine-tunées avec chaque tête"
                 + (f" ; couche {final_layer + 1 if final_layer == 31 else f'{final_layer + 1}–32'} "
                    f"dégelée mais jamais lue par la tête (aucun gradient → poids pré-entraînés, "
                    f"identiques entre modèles)" if final_layer < 32 else "")
                 + ".\n")
        if frozen_check is not None:
            status = "OK" if frozen_check < 1e-3 else "⚠ écart inattendu, vérifier le chargement des checkpoints"
            L.append(f"Contrôle de cohérence — écart max des cartes d'attention sur les couches gelées "
                     f"entre modèles : {frozen_check:.2e} ({status}).\n")
        # Colonne de référence : dernière couche gelée, ou couche 1 si tout est fine-tuné
        ref_col = max(ref_layer, 1)
        L.append("| Modèle | Distance moy. couche "
                 f"{ref_col} (px) | Distance moy. couche {final_layer} (px) | Lift sémantique couche "
                 f"{ref_col} | Lift sémantique couche {final_layer} | Masse même classe au point requête "
                 f"(couche {final_layer}) |")
        L.append("|---|---|---|---|---|---|")
        for s in specs:
            a = results[s.name]["attn"]
            q = [r["same_class_mass"] for r in a["query_rows"] if r["layer"] == final_layer]
            q = np.array(q)[~np.isnan(q)]
            L.append(f"| {s.label} | {a['mean_distance'][ref_col - 1].mean():.1f} | "
                     f"{a['mean_distance'][final_layer - 1].mean():.1f} | "
                     f"{a['semantic_lift'][ref_col - 1].mean():.2f} | "
                     f"{a['semantic_lift'][final_layer - 1].mean():.2f} | "
                     f"{np.mean(q):.3f} ± {np.std(q) / np.sqrt(max(len(q), 1)):.3f} (s.e.) |")
        L.append("\n![Stats d'attention](attention/layer_stats.png)\n")
        L.append("Cartes : `attention/attn_img<idx>.png` (toutes les requêtes d'une image, couche gelée "
                 "de référence vs dernière couche de chaque modèle) et `attention/attn_img<idx>_q<j>_layers.png` "
                 "(évolution par couche). Échelle : attention × N (1 = uniforme), même échelle par ligne.\n")

    L.append("## Limites\n")
    L.append("- Une seule seed d'entraînement par modèle : les IC ne capturent que la variabilité due à "
             "l'échantillon d'images, pas la variance d'entraînement.")
    if args.ckpt_name == "best.pth":
        L.append("- `best.pth` est sélectionné sur ce même split val → biais optimiste (léger, et d'autant "
                 "plus grand que les validations ont été fréquentes). `--ckpt-name last.pth` l'évite.")
    notes = [f"{s.label} : {s.note}" for s in specs if s.note]
    if notes:
        L.append("- Écarts de protocole d'entraînement : " + " ; ".join(notes) + ".")
    L.append("- Les cartes d'attention décrivent le backbone, pas la tête : elles ne diffèrent entre modèles "
             f"que par les blocs fine-tunés ({ref_layer + 1}–{final_layer}), co-adaptés à chaque tête.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dirs", default="work_dirs")
    prefixes = [c[0] for c in MODEL_CONFIGS]
    parser.add_argument("--run-suffix", default="512x512",
                        help="Suffixe des dossiers de run dans --work-dirs : <préfixe><suffixe> "
                             "(ex. '512x512' → simple512x512, '_fullres' → simple_fullres)")
    parser.add_argument("--models", nargs="+", default=prefixes, choices=prefixes)
    parser.add_argument("--baseline", default=None, choices=prefixes,
                        help="Modèle de référence pour la figure des Δ IoU (défaut : le premier)")
    parser.add_argument("--ckpt-name", default="best.pth")
    parser.add_argument("--output-dir", default="results/comparison_512x512")
    parser.add_argument("--model-name", default="facebook/ijepa_vith14_22k")
    parser.add_argument("--layer-indices", type=int, nargs=4, default=[7, 15, 23, 31])
    parser.add_argument("--trained-unfreeze-last-n", type=int, default=4,
                        help="Nombre de blocs fine-tunés à l'entraînement (contrôle + annotations)")
    # --- protocole d'inférence ---
    parser.add_argument("--eval-mode", default="slide", choices=["slide", "whole"])
    parser.add_argument("--crop-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--stride", type=int, nargs=2, default=[341, 341])
    parser.add_argument("--window-batch", type=int, default=6, help="Fenêtres traitées par forward")
    parser.add_argument("--amp", action="store_true", help="Inférence en fp16 (défaut : fp32)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None, help="Limiter (debug)")
    # --- statistiques / visualisation ---
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-vis", type=int, default=8, help="Images pour les figures qualitatives")
    parser.add_argument("--vis-indices", type=int, nargs="+", default=None,
                        help="Indices val explicites (remplace le tirage aléatoire)")
    # --- attention ---
    parser.add_argument("--num-attn-images", type=int, default=25,
                        help="Images utilisées pour les statistiques d'attention (0 = désactivé)")
    parser.add_argument("--num-attn-figures", type=int, default=4, help="Images avec figures d'attention")
    parser.add_argument("--attn-queries", type=int, default=3, help="Points requêtes max par image")
    parser.add_argument("--attn-min-area", type=int, default=1500,
                        help="Aire min (px) de l'objet GT pour y placer une requête")
    parser.add_argument("--attn-layers", type=int, nargs="+", default=[16, 28, 30, 31],
                        help="Couches (1..32) affichées dans les figures d'attention")
    parser.add_argument("--attn-crop-size", type=int, nargs=2, default=None,
                        help="Taille des fenêtres d'attention (défaut : --crop-size, i.e. ce que le "
                             "modèle voit en inférence)")
    parser.add_argument("--attn-chunk", type=int, default=1024,
                        help="Lignes de la matrice d'attention calculées à la fois (mémoire GPU)")
    parser.add_argument("--no-cache", action="store_true", help="Ignorer le cache et tout ré-évaluer")
    args = parser.parse_args()

    # Classes absentes d'un sous-ensemble / d'une réplique bootstrap → NaN attendus
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(NaN|empty slice).*")
    accelerator = Accelerator()
    device = accelerator.device
    global log
    log = accelerator.print
    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 strict par défaut (reproductibilité)
    torch.backends.cudnn.allow_tf32 = False

    global MODEL_ZOO
    MODEL_ZOO = make_model_zoo(args.run_suffix)
    specs = [MODEL_ZOO[m + args.run_suffix] for m in args.models]
    baseline = MODEL_ZOO[args.baseline + args.run_suffix] if args.baseline else specs[0]
    if args.attn_crop_size is None:
        args.attn_crop_size = list(args.crop_size)
    out = args.output_dir
    for sub in ("", "cache", "qualitative", "predictions", "attention"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    dataset = CityscapesSegDataset(args.data_root, split="val", train=False)
    n_images = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))

    rng = np.random.default_rng(args.seed)
    if args.vis_indices:
        vis = sorted(i for i in args.vis_indices if i < n_images)
    else:
        vis = sorted(rng.choice(n_images, size=min(args.num_vis, n_images), replace=False).tolist())
    extra = [i for i in rng.permutation(n_images).tolist() if i not in vis]
    attn_indices = (vis + extra)[:args.num_attn_images] if args.num_attn_images > 0 else []

    # Dernière couche gelée (0 si tout le backbone a été fine-tuné → pas de référence commune)
    ref_layer = max(32 - args.trained_unfreeze_last_n, 0)
    # Dernière couche LUE par la tête (hidden_states[31]). Le bloc 32 est dégelé par
    # --unfreeze-last-n 4 mais ne reçoit jamais de gradient : ses poids sont identiques
    # au pré-entraînement dans les 4 checkpoints, donc sans intérêt pour la comparaison.
    final_layer = max(args.layer_indices)
    args.attn_layers = sorted((set(args.attn_layers) | {ref_layer, final_layer}) - {0})

    patch = 14
    attn_windows, grid = build_attention_windows(dataset, attn_indices, args, patch)
    log(f"{n_images} images val | visualisation : {vis} | "
          f"{len(attn_windows)} fenêtres d'attention sur {len(attn_indices)} images")

    # ---------------- évaluation modèle par modèle ----------------
    results = {}
    for spec in specs:
        cache_path = os.path.join(out, "cache", f"{spec.name}.pt")
        sig = cache_signature(spec, args, n_images, vis, attn_windows)
        if not args.no_cache and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("signature") == sig:
                log(f"\n=== {spec.label} ({spec.name}) : résultats en cache")
                results[spec.name] = cached["result"]
                continue
        log(f"\n=== {spec.label} ({spec.name})")
        t0 = time.time()
        results[spec.name] = evaluate_model(spec, args, dataset, n_images, vis, attn_windows, grid, accelerator)
        log(f"  terminé en {(time.time() - t0) / 60:.1f} min")
        if accelerator.is_main_process:
            torch.save({"signature": sig, "result": results[spec.name]}, cache_path)
        accelerator.wait_for_everyone()

    # Statistiques, figures et exports : processus principal uniquement
    if not accelerator.is_main_process:
        return

    # ---------------- statistiques ----------------
    stats, pairs = compute_statistics(results, [s.name for s in specs], args)

    log("\n" + "=" * 78)
    log(f"{'Modèle':24s} {'mIoU':>7s}  {'IC 95 %':>16s} {'mAcc':>7s} {'aAcc':>7s} {'catIoU':>7s}")
    for s in specs:
        st = stats[s.name]
        log(f"{s.label:24s} {100 * st['miou']:7.2f}  [{100 * st['miou_ci'][0]:6.2f}, "
              f"{100 * st['miou_ci'][1]:6.2f}] {100 * st['macc']:7.2f} {100 * st['aacc']:7.2f} "
              f"{100 * st['cat_miou']:7.2f}")
    log("\nIoU par classe :")
    log(f"{'':15s}" + "".join(f"{s.label[:18]:>20s}" for s in specs))
    for c, name in enumerate(CLASS_NAMES):
        log(f"{name:15s}" + "".join(f"{100 * stats[s.name]['iou'][c]:20.2f}" for s in specs))
    for p in pairs:
        log(f"Δ {MODEL_ZOO[p['b']].label} − {MODEL_ZOO[p['a']].label} : {100 * p['delta']:+.2f} pts "
              f"[{100 * p['ci'][0]:+.2f}, {100 * p['ci'][1]:+.2f}]  p_holm={p['p_holm']:.3f}")

    # ---------------- figures quantitatives ----------------
    fig_miou_ci(stats, specs, os.path.join(out, "miou_ci.png"))
    fig_per_class_iou(stats, specs, os.path.join(out, "per_class_iou.png"))
    fig_delta_heatmap(stats, specs, baseline, os.path.join(out, "per_class_delta.png"))
    fig_confusions(stats, specs, os.path.join(out, "confusion_matrices.png"))

    # ---------------- qualitatif ----------------
    images, gts = {}, {}
    for idx in vis:
        image_t, label_t = dataset[idx]
        images[idx], gts[idx] = denormalize(image_t), label_t.numpy()
        preds = {s.name: results[s.name]["vis_preds"][idx] for s in specs}
        fig_qualitative(idx, images[idx], gts[idx], preds, specs, stats,
                        os.path.join(out, "qualitative", f"img{idx:03d}.png"))
        cv2.imwrite(os.path.join(out, "predictions", f"{idx:03d}_image.png"),
                    cv2.cvtColor((images[idx] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out, "predictions", f"{idx:03d}_gt.png"),
                    cv2.cvtColor(colorize(gts[idx]), cv2.COLOR_RGB2BGR))
        for s in specs:
            cv2.imwrite(os.path.join(out, "predictions", f"{idx:03d}_{s.name}.png"),
                        cv2.cvtColor(colorize(preds[s.name]), cv2.COLOR_RGB2BGR))
    fig_qualitative_overview(vis, images, gts, results, specs, os.path.join(out, "qualitative", "overview.png"))

    # ---------------- attention ----------------
    frozen_check = None
    if attn_windows:
        ref_maps = results[specs[0].name]["attn"]["maps"]
        diffs = [np.abs(results[s.name]["attn"]["maps"][k] - v).max()
                 for s in specs[1:] for k, v in ref_maps.items()
                 if k[2] <= ref_layer or k[2] > final_layer]
        frozen_check = float(max(diffs)) if diffs else None   # None : aucune couche commune à comparer
        fig_attention(attn_windows, results, specs, args, grid, patch,
                      os.path.join(out, "attention"), ref_layer, final_layer)
        fig_attention_layer_stats(results, specs, ref_layer, final_layer, os.path.join(out, "attention", "layer_stats.png"))

    # ---------------- exports ----------------
    write_csvs(stats, specs, pairs, results, out)
    write_latex(stats, specs, os.path.join(out, "per_class_iou.tex"))
    write_report(stats, specs, pairs, results, args, n_images, vis, attn_windows, ref_layer,
                 final_layer, frozen_check, os.path.join(out, "report.md"))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "models": {s.name: {
                "label": s.label, "info": results[s.name]["info"],
                "miou": stats[s.name]["miou"], "miou_ci": stats[s.name]["miou_ci"].tolist(),
                "macc": stats[s.name]["macc"], "aacc": stats[s.name]["aacc"],
                "fwiou": stats[s.name]["fwiou"], "category_miou": stats[s.name]["cat_miou"],
                "iou": dict(zip(CLASS_NAMES, np.nan_to_num(stats[s.name]["iou"], nan=-1).tolist())),
                "latency_ms": results[s.name]["latency_ms"],
            } for s in specs},
            "pairwise": [{**p, "ci": p["ci"].tolist()} for p in pairs],
            "frozen_layer_attention_max_diff": frozen_check,
        }, f, indent=2, ensure_ascii=False)
    log(f"\nRapport : {os.path.join(out, 'report.md')}")


if __name__ == "__main__":
    main()
