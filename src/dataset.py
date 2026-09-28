"""
Dataset Cityscapes avec les transformations alignées sur la config MMSegmentation
de référence (pipeline train + test identiques).
"""

import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

# Mapping officiel Cityscapes labelId -> trainId
_LABELID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18, -1: 255,
}
_LUT = np.array([_LABELID_TO_TRAINID.get(i, 255) for i in range(-1, 34)], dtype=np.uint8)

# Identiques aux valeurs ImageNet (config MMSeg en base 0-255, ici en [0,1])
# 123.675/255 ≈ 0.485, 116.28/255 ≈ 0.456, 103.53/255 ≈ 0.406
CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD  = [0.229, 0.224, 0.225]


def labelid_to_trainid(label_img: Image.Image) -> np.ndarray:
    arr = np.array(label_img, dtype=np.int64)
    return _LUT[arr + 1]


def photo_metric_distortion(image: Image.Image) -> Image.Image:
    """
    Reproduit le PhotoMetricDistortion de MMSeg :
      brightness → contrast (mode aléatoire) → saturation → hue → contrast (mode aléatoire)
    Tous les facteurs sont tirés uniformément dans [0.5, 1.5],
    sauf la teinte dans [-18°, +18°].
    """
    # Brightness
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        image = ImageEnhance.Brightness(image).enhance(factor)

    # Contrast – mode 0 : avant saturation/hue
    contrast_mode = random.randint(0, 1)
    if contrast_mode == 0 and random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        image = ImageEnhance.Contrast(image).enhance(factor)

    # Saturation
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        image = ImageEnhance.Color(image).enhance(factor)

    # Hue  (PIL attend une valeur en [-0.5, 0.5], MMSeg utilise ±18° → ±18/360)
    if random.random() < 0.5:
        hue_delta = random.uniform(-18 / 360, 18 / 360)
        image = TF.adjust_hue(image, hue_delta)

    # Contrast – mode 1 : après saturation/hue
    if contrast_mode == 1 and random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        image = ImageEnhance.Contrast(image).enhance(factor)

    return image

def resize_keep_ratio(image: Image.Image, label: np.ndarray,
                      target_hw: tuple) -> tuple:
    """
    Redimensionne image + label pour que la grande dimension corresponde à
    target_hw en conservant le ratio (comme Resize keep_ratio=True de MMSeg).
    Sur Cityscapes (2048×1024) avec target (2048, 1024) → pas de changement.
    """
    target_h, target_w = target_hw
    orig_w, orig_h = image.size          # PIL : (w, h)
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale + 0.5)
    new_h = int(orig_h * scale + 0.5)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    label = np.array(
        Image.fromarray(label).resize((new_w, new_h), Image.NEAREST),
        dtype=np.int64,
    )
    return image, label


class CityscapesSegDataset(Dataset):
    """
    Wrapper Cityscapes aligné sur le pipeline MMSeg de référence :

    Train  : Resize (ratio 0.5-2.0) → RandomCrop (512×1024) → RandomFlip (p=0.5)
             → PhotoMetricDistortion → Normalize → Pad si nécessaire
    Val/test: Resize keep_ratio → (2048×1024) → Normalize
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        # CORRECTION 1 : crop_size = (512, 1024) comme dans la config
        crop_size: tuple = (512, 1024),
        train: bool = True,
    ):
        self.base = Cityscapes(root, split=split, mode="fine", target_type="semantic")
        self.crop_size = crop_size   # (H, W)
        self.train = train

    def __len__(self):
        return len(self.base)

    # ------------------------------------------------------------------
    # Pipeline TRAIN
    # ------------------------------------------------------------------
    def _augment_train(self, image: Image.Image, label: np.ndarray):
        # --- Resize avec ratio aléatoire ---
        # CORRECTION 2 : ratio_range = (0.5, 2.0) et non (0.5, 1.5)
        scale = random.uniform(0.5, 2.0)
        new_w = int(image.width  * scale + 0.5)
        new_h = int(image.height * scale + 0.5)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        label = np.array(
            Image.fromarray(label).resize((new_w, new_h), Image.NEAREST),
            dtype=np.int64,
        )

        # --- Pad AVANT le crop si l'image est trop petite (seg_pad_val=255) ---
        # CORRECTION 3 : le pad est appliqué avant le crop, comme dans MMSeg
        crop_h, crop_w = self.crop_size
        pad_bottom = max(crop_h - new_h, 0)
        pad_right  = max(crop_w - new_w, 0)
        if pad_bottom > 0 or pad_right > 0:
            image = TF.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
            label_pil = Image.fromarray(label.astype(np.uint8))
            label_pil = TF.pad(label_pil, [0, 0, pad_right, pad_bottom], fill=255)
            label = np.array(label_pil, dtype=np.int64)

        # --- RandomCrop (512 × 1024) ---
        image, label = self._random_crop_with_cat_ratio(image, label, crop_h, crop_w)

        # --- RandomFlip horizontal (prob=0.5) ---
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = label[:, ::-1].copy()

        # CORRECTION 4 : PhotoMetricDistortion ajouté
        image = photo_metric_distortion(image)

        return image, label

    # ------------------------------------------------------------------
    # Pipeline VAL / TEST
    # ------------------------------------------------------------------
    def _preprocess_val(self, image: Image.Image, label: np.ndarray):
        # CORRECTION 5 : resize keep_ratio vers (2048, 1024) comme MultiScaleFlipAug
        # (flip=False → pas de flip, pas d'augmentation)
        image, label = resize_keep_ratio(image, label, target_hw=(1024, 2048))
        return image, label

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        image, label_img = self.base[idx]
        label = labelid_to_trainid(label_img)

        if self.train:
            image, label = self._augment_train(image, label)
        else:
            image, label = self._preprocess_val(image, label)

        # Normalize (valeurs équivalentes à la config : 123.675/255 = 0.485, etc.)
        image_t = TF.to_tensor(image)                                    # [0, 1]
        image_t = TF.normalize(image_t, mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)
        label_t = torch.from_numpy(label).long()

        return image_t, label_t
    
    def _random_crop_with_cat_ratio(self, image, label, crop_h, crop_w,
                                    cat_max_ratio=0.75, max_tries=10):
        img_w, img_h = image.size
        for _ in range(max_tries):
            x = random.randint(0, img_w - crop_w)
            y = random.randint(0, img_h - crop_h)
            crop_label = label[y : y + crop_h, x : x + crop_w]
            # Ignore la classe 255 dans le calcul
            valid = crop_label[crop_label != 255]
            if valid.size == 0:
                break
            counts = np.bincount(valid.astype(np.int64), minlength=19)
            if counts.max() / valid.size < cat_max_ratio:
                break  # crop acceptable
        image = image.crop((x, y, x + crop_w, y + crop_h))
        label = crop_label
        return image, label
