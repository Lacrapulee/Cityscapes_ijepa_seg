#!/bin/bash
# Télécharge (ou vérifie le cache) du checkpoint I-JEPA ViT-H/14 depuis HuggingFace.
#
# Avec l'approche pure PyTorch (AutoModel.from_pretrained), le téléchargement se fait
# normalement automatiquement au premier lancement de train.py -- ce script sert
# surtout à pré-télécharger le modèle une fois, pour ne pas dépendre du réseau
# pendant l'entraînement, et pour vérifier que tout fonctionne avant de lancer un run long.

set -e

MODEL_NAME="${1:-facebook/ijepa_vith14_22k}"

echo "Vérification des dépendances..."
python -c "import transformers, huggingface_hub" 2>/dev/null || {
    echo "Erreur : installez d'abord les dépendances avec :"
    echo "  pip install transformers huggingface_hub"
    exit 1
}

echo "Téléchargement de ${MODEL_NAME} (peut prendre plusieurs minutes, ~2.5 Go)..."
python - <<EOF
from transformers import AutoModel, AutoImageProcessor

model_name = "${MODEL_NAME}"
model = AutoModel.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

n_params = sum(p.numel() for p in model.parameters())
print(f"Modèle chargé : {model_name}")
print(f"Nombre de paramètres : {n_params / 1e6:.1f}M")
print(f"Dimension des embeddings : {model.config.hidden_size}")
print(f"Nombre de couches : {model.config.num_hidden_layers}")
print(f"Taille de patch : {model.config.patch_size}")
EOF

echo ""
echo "OK. Le modèle est en cache local (~/.cache/huggingface) et sera rechargé"
echo "automatiquement par train.py et test.py sans re-téléchargement."
