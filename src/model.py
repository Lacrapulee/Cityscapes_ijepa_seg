"""
Modèle de segmentation : encodeur I-JEPA (gelé) + tête DPT (Reassemble + Fusion,
Ranftl et al. 2021, "Vision Transformers for Dense Prediction").

Aucune compilation CUDA custom nécessaire — tout est en PyTorch/torchvision standard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from dpt.blocks import FeatureFusionBlock_custom


class IJepaEncoder(nn.Module):
    """
    Encapsule le ViT-H I-JEPA pré-entraîné (HuggingFace).
    Renvoie les features de plusieurs profondeurs (couches), reshapées en grille spatiale.

    Par défaut entièrement gelé (unfreeze_last_n=0), comme avant. Si unfreeze_last_n > 0,
    les N derniers blocs transformer sont rendus entraînables (fine-tuning partiel) --
    le reste du backbone (couches précoces + patch embedding) reste gelé.
    """

    def __init__(
        self,
        model_name: str = "facebook/ijepa_vith14_22k",
        layer_indices: tuple = (7, 15, 23, 31),
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.layer_indices = layer_indices
        self.embed_dim = self.encoder.config.hidden_size
        self.patch_size = self.encoder.config.patch_size
        self.unfreeze_last_n = unfreeze_last_n

        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0:
            transformer_layers = self._get_transformer_layers()
            n_layers = len(transformer_layers)
            if unfreeze_last_n > n_layers:
                raise ValueError(
                    f"unfreeze_last_n={unfreeze_last_n} > nombre de couches disponibles ({n_layers})"
                )
            for layer in transformer_layers[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()

        self._sync_mode()

    def _get_transformer_layers(self):
        for attr_name in ("layer", "layers"):
            if hasattr(self.encoder, attr_name):
                return getattr(self.encoder, attr_name)
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, attr_name):
                return getattr(self.encoder.encoder, attr_name)
        raise AttributeError(
            "Impossible de localiser les blocs transformer d'I-JEPA automatiquement. "
            "Lancez `print(model.encoder.encoder)` (ou `print(model.encoder)`) pour "
            "trouver le nom exact de l'attribut contenant le ModuleList des couches, "
            "et ajustez `_get_transformer_layers` en conséquence."
        )

    def _sync_mode(self):
        if self.unfreeze_last_n > 0:
            self.encoder.train()
        else:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._sync_mode()
        return self

    def forward(self, pixel_values: torch.Tensor):
        B, C, H, W = pixel_values.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        forward_kwargs = dict(output_hidden_states=True, interpolate_pos_encoding=True)

        if self.unfreeze_last_n > 0:
            outputs = self.encoder(pixel_values, **forward_kwargs)
        else:
            with torch.no_grad():
                outputs = self.encoder(pixel_values, **forward_kwargs)

        hidden_states = outputs.hidden_states

        features = []
        for idx in self.layer_indices:
            tokens = hidden_states[idx]
            feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
            features.append(feat)
        return features


class ReassembleStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale_factor: float):
        super().__init__()
        self.project = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, feature_map: torch.Tensor):
        x = self.project(feature_map)
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return x


class SafeFeatureFusionBlock(FeatureFusionBlock_custom):
    """
    FeatureFusionBlock_custom officiel (isl-org/DPT, src/dpt/blocks.py), avec un
    garde-fou avant l'addition skip.

    Le code officiel fait `output + resConfUnit1(skip)` en supposant que les deux
    tenseurs ont toujours la même taille spatiale. C'est vrai pour un ViT/16 en
    384×384 (grille de patches = 24, toujours paire), mais pas pour I-JEPA en
    patch_size=14 sur les résolutions utilisées ici (ex. 1024×2048 en val →
    grille 73×146, 73 est impair) : après un cycle downscale/upscale, `path` et
    le `skip` réassemblé peuvent différer d'un pixel, ce qui fait planter
    l'addition (cf. le crash "72 vs 73" observé avec ReassembleStage). On
    réaligne donc le skip par interpolation explicite sur la taille exacte du
    path courant avant de déléguer au forward officiel — c'est exactement ce
    que fait transformers.models.dpt.modeling_dpt.DPTFeatureFusionLayer côté
    HuggingFace.
    """

    def forward(self, *xs):
        if len(xs) == 2 and xs[0].shape[-2:] != xs[1].shape[-2:]:
            output, skip = xs
            skip = F.interpolate(
                skip, size=output.shape[-2:], mode="bilinear", align_corners=False
            )
            xs = (output, skip)
        return super().forward(*xs)


def _make_fusion_block(channels: int, use_bn: bool = False) -> SafeFeatureFusionBlock:
    return SafeFeatureFusionBlock(
        channels,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class MultiDepthFusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.reduce = nn.ModuleList(
            [nn.Conv2d(in_dim, out_dim, kernel_size=1) for _ in range(n_layers)]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * n_layers, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list):
        reduced = [conv(f) for conv, f in zip(self.reduce, features)]
        fused = torch.cat(reduced, dim=1)
        return self.fuse(fused)


class ProgressiveUpsampleDecoder(nn.Module):
    def __init__(self, in_dim: int = 256, num_classes: int = 19, n_upsample_steps: int = 3):
        super().__init__()
        blocks = []
        dim = in_dim
        for _ in range(n_upsample_steps):
            next_dim = max(dim // 2, 32)
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dim, next_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(next_dim),
                    nn.ReLU(inplace=True),
                )
            )
            dim = next_dim
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: tuple):
        x = self.blocks(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.classifier(x)


class SimpleHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 19,
        channels: int = 256,
        n_layers: int = 4,
        fusion_type: str = "multidepth",
        scale_factors: tuple = (4, 2, 1, 0.5),
        use_bn: bool = False,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "multidepth":
            self.fusion = MultiDepthFusion(in_dim=in_dim, out_dim=channels, n_layers=n_layers)
            # MultiDepthFusion reste à la résolution de base de la grille de patches :
            # il lui faut ces 3 upsamplings progressifs pour atteindre une résolution
            # exploitable avant l'interpolation finale vers output_size.
            self.decoder = ProgressiveUpsampleDecoder(in_dim=channels, num_classes=num_classes)

        elif fusion_type == "feature":
            self.reassemble = nn.ModuleList(
                [ReassembleStage(in_dim, channels, s) for s in scale_factors]
            )
            self.fusion = nn.ModuleList(
                [_make_fusion_block(channels, use_bn=use_bn) for _ in scale_factors]
            )
            # La pyramide Reassemble/Fusion fait déjà monter la résolution à ~8x la
            # grille de patches de base (cf. scale_factors + le x2 de chaque étage de
            # fusion) : lui appliquer en plus ProgressiveUpsampleDecoder (3 upsamplings
            # x2, donc x8 de plus) fait exploser la mémoire pour rien avant que
            # l'interpolation finale ne rejette tout ce surplus de résolution. On
            # utilise donc une tête légère, comme DPTHead pour ce même fusion_type.
            self.output_head = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, num_classes, kernel_size=1),
            )

        else:
            raise ValueError(f"fusion_type doit être 'multidepth' ou 'feature', reçu : {fusion_type!r}")

    def forward(self, features: list, output_size: tuple):
        if self.fusion_type == "feature":
            reassembled = [stage(f) for stage, f in zip(self.reassemble, features)]

            path = self.fusion[-1](reassembled[-1])
            for i in range(len(reassembled) - 2, -1, -1):
                path = self.fusion[i](path, reassembled[i])

            logits = self.output_head(path)
            return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        elif self.fusion_type == "multidepth":
            fused = self.fusion(features)
            return self.decoder(fused, output_size)

 #
class DPTHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 19,
        channels: int = 256,
        scale_factors: tuple = (4, 2, 1, 0.5),
        fusion_type: str = "feature",
        n_layers: int = 4,
        use_bn: bool = False,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "multidepth":
            self.fusion = MultiDepthFusion(in_dim=in_dim, out_dim=channels, n_layers=n_layers)
        elif fusion_type == "feature":
            self.reassemble = nn.ModuleList(
                [ReassembleStage(in_dim, channels, s) for s in scale_factors]
            )
            self.fusion = nn.ModuleList(
                [_make_fusion_block(channels, use_bn=use_bn) for _ in scale_factors]
            )
        else:
            raise ValueError(f"fusion_type doit être 'multidepth' ou 'feature', reçu : {fusion_type!r}")

        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, features: list, output_size: tuple):
        if self.fusion_type == "feature":
            reassembled = [stage(f) for stage, f in zip(self.reassemble, features)]

            path = self.fusion[-1](reassembled[-1])
            for i in range(len(reassembled) - 2, -1, -1):
                path = self.fusion[i](path, reassembled[i])

        elif self.fusion_type == "multidepth":
            path = self.fusion(features)

        logits = self.output_head(path)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class IJepaSegmentationModel(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/ijepa_vith14_22k",
        num_classes: int = 19,
        layer_indices: tuple = (7, 15, 23, 31),
        decoder_dim: int = 256,
        unfreeze_last_n: int = 0,
        decoder_type: str = "dpt",
        fusion_type: str = "feature",
        scale_factors: tuple = (4, 2, 1, 0.5),
        use_bn: bool = False,
    ):
        super().__init__()
        self.encoder = IJepaEncoder(model_name, layer_indices, unfreeze_last_n=unfreeze_last_n)

        n_layers = len(layer_indices)

        if decoder_type == "dpt":
            self.head = DPTHead(
                in_dim=self.encoder.embed_dim,
                num_classes=num_classes,
                channels=decoder_dim,
                scale_factors=scale_factors,
                fusion_type=fusion_type,
                n_layers=n_layers,
                use_bn=use_bn,
            )
        elif decoder_type == "simple":
            self.head = SimpleHead(
                in_dim=self.encoder.embed_dim,
                num_classes=num_classes,
                channels=decoder_dim,
                n_layers=n_layers,
                fusion_type=fusion_type,
                scale_factors=scale_factors,
                use_bn=use_bn,
            )
        else:
            raise ValueError(f"decoder_type doit être 'dpt' ou 'simple', reçu : {decoder_type!r}")

    def forward(self, pixel_values: torch.Tensor):
        H, W = pixel_values.shape[-2:]
        features = self.encoder(pixel_values)
        return self.head(features, output_size=(H, W))

    def get_trainable_backbone_state_dict(self):
        return {
            name: p.detach().cpu()
            for name, p in self.encoder.named_parameters()
            if p.requires_grad
        }

    def load_trainable_backbone_state_dict(self, partial_state_dict: dict):
        if not partial_state_dict:
            return
        full_state = self.encoder.state_dict()
        full_state.update(partial_state_dict)
        self.encoder.load_state_dict(full_state)

    def param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        groups = [{"params": list(self.head.parameters()), "lr": base_lr}]
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult})
        return groups