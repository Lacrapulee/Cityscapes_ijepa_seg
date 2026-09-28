import torch
from torchinfo import summary
from model import IJepaSegmentationModel

# Charger le modèle complet 
model = IJepaSegmentationModel(
        model_name="facebook/ijepa_vith14_22k",
        num_classes=19,
        layer_indices=[7, 15, 23, 31],
        unfreeze_last_n=0,
    )
# Toujours passer le modèle en mode évaluation pour l'inférence/inspection

summary(model, input_size=(1, 3, 224, 224))