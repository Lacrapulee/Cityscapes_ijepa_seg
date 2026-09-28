#!/usr/bin/env bash
# Entraîne les 4 configurations tête/fusion, backbone I-JEPA entièrement dégelé,
# crops pleine résolution 1024×2048, batch GLOBAL = 8 quel que soit le nombre de GPU.
#
# Usage (dans le conteneur, depuis /workspace) :
#   bash scripts/train_fullres.sh
#   BATCH_PER_GPU=2 bash scripts/train_fullres.sh          # GPU 80 Go : moins d'accumulation
#   CONFIGS="dpt simple" EPOCHS=1 bash scripts/train_fullres.sh   # test rapide
#
# Variables : NGPU (défaut : tous les GPU visibles), BATCH_PER_GPU (défaut 1),
# GLOBAL_BATCH (défaut 8), EPOCHS (50), MIXED_PRECISION (bf16 | fp16 | no), CONFIGS, SUFFIX.
set -euo pipefail

NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
BATCH_PER_GPU=${BATCH_PER_GPU:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-8}
EPOCHS=${EPOCHS:-50}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
CONFIGS=${CONFIGS:-"simple simplefeature dpt dptmultidepth"}
SUFFIX=${SUFFIX:-_fullres}
DATA_ROOT=${DATA_ROOT:-data/cityscapes}

per_step=$((NGPU * BATCH_PER_GPU))
if (( per_step > GLOBAL_BATCH || GLOBAL_BATCH % per_step != 0 )); then
    echo "Batch global ${GLOBAL_BATCH} non divisible par NGPU×BATCH_PER_GPU = ${per_step}" >&2
    exit 1
fi
ACCUM=$((GLOBAL_BATCH / per_step))

launch=(accelerate launch --num_processes "$NGPU" --mixed_precision "$MIXED_PRECISION")
(( NGPU > 1 )) && launch+=(--multi_gpu)

echo "${NGPU} GPU × batch ${BATCH_PER_GPU} × accumulation ${ACCUM} = batch global ${GLOBAL_BATCH} | ${MIXED_PRECISION}"

for cfg in $CONFIGS; do
    case $cfg in
        simple)        dec=simple; fus=multidepth ;;
        simplefeature) dec=simple; fus=feature ;;
        dpt)           dec=dpt;    fus=feature ;;
        dptmultidepth) dec=dpt;    fus=multidepth ;;
        *) echo "Configuration inconnue : $cfg" >&2; exit 1 ;;
    esac
    run="${cfg}${SUFFIX}"
    echo "=== ${run} (${dec} / ${fus})"
    "${launch[@]}" src/train.py \
        --data-root "$DATA_ROOT" \
        --output-dir "work_dirs/${run}" \
        --decoder-type "$dec" --fusion-type "$fus" \
        --crop-size 1024 2048 \
        --unfreeze-last-n 32 \
        --batch-size "$BATCH_PER_GPU" \
        --gradient-accumulation-steps "$ACCUM" \
        --epochs "$EPOCHS" \
        --val-interval 5 \
        --num-workers 8 \
        --seed 42 \
        --wandb --run-name "$run"
done
