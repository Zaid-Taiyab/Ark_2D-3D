#!/bin/bash
# ======================================================================================================
#  Learning-rate range test for the Ark+ baselines (lr_finder.py, branch switchable-3d code).
#  One fresh model per dataset; lr rises exponentially 1e-5 -> 1 over 200 steps; stops when the loss explodes.
#  Same settings as the real runs: unified Swin-B (identical to Swin-B on X-rays), ImageNet init, --opt sgd,
#  projector 1376, CT batch 8 from the yaml with bf16, 2D batch 200 in fp32.
#  Output: /scratch/ztaiyab/ark_runs/lr_finder/lr_finder_<dataset>.csv  (+ log). Plot afterwards with
#  plot_lr_finder.py (see the end of the log for the command).
# ======================================================================================================
#SBATCH --job-name=ark_lr_finder
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_imaging_spring2026
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ztaiyab/ark_runs/lr_finder/lr_finder_%j.out
#SBATCH --error=/scratch/ztaiyab/ark_runs/lr_finder/lr_finder_%j.err

set -eo pipefail
OUT=/scratch/ztaiyab/ark_runs/lr_finder
SIF=/scratch/ztaiyab/ark_runs/containers/ark_vindr.sif
CODE_DIR=/scratch/ztaiyab/ark_code/Ark_unified/Ark_Plus/Pretraining
YAML=/scratch/ztaiyab/ark_runs/configs/datasets_config_2d3d.yaml
DATA_LISTS=/scratch/ztaiyab/ark_code/Ark/Ark_Plus/dataset
WEIGHTS=/scratch/ztaiyab/weights/swin_base_patch4_window7_224_22kto1k.pth
export PYTHONNOUSERSITE=1
RUN="apptainer exec --nv --env WANDB_MODE=disabled --bind /scratch/ztaiyab:/scratch/ztaiyab $SIF"

for p in "$CODE_DIR/lr_finder.py" "$YAML" "$DATA_LISTS" "$WEIGHTS" "$SIF"; do [ -e "$p" ] || { echo "ERROR: missing $p"; exit 1; }; done
mkdir -p "$OUT" && cd "$OUT"
cp "$YAML" datasets_config.yaml
ln -sfn "$DATA_LISTS" dataset
echo "git: $(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD) $(git -C "$CODE_DIR" rev-parse --short HEAD)"

COMMON="--model unified_swin_base --init imagenet --pretrained_weights $WEIGHTS --opt sgd --batch_size 200 \
        --projector_features 1376 --input_size 224 --img_resize 256 --window_depth 4 --patch_depth 4 \
        --grad_checkpoint True --workers 12 --finder_min 1e-5 --finder_max 1 --finder_iters 200"
for ds in RICORD LUNA16; do
  echo "================ $ds (3D, bf16)"
  $RUN python "$CODE_DIR/lr_finder.py" --data_set $ds --amp True $COMMON --finder_out "$OUT/lr_finder_$ds.csv" || echo "!! $ds failed"
done
for ds in ChestXray14 VinDrCXR; do
  echo "================ $ds (2D, fp32)"
  $RUN python "$CODE_DIR/lr_finder.py" --data_set $ds --amp False $COMMON --finder_out "$OUT/lr_finder_$ds.csv" || echo "!! $ds failed"
done
echo "done. Plot on the login node with:"
echo "  cd $OUT && python $CODE_DIR/plot_lr_finder.py lr_finder_RICORD.csv lr_finder_LUNA16.csv lr_finder_ChestXray14.csv lr_finder_VinDrCXR.csv --out $OUT"
