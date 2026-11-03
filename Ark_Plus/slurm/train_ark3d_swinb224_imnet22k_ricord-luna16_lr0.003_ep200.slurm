#!/bin/bash
# ======================================================================================================
#  3D baseline | unified 2D/3D Swin-B used on CT only (branch switchable-3d) | ImageNet-22k->1k inflated to 3D
#  Datasets : RICORD + LUNA16    (no X-ray data, no DRR heads: drr_head/return_drr false in the yaml)
#  Same recipe as the 2D arkfix runs: SGD, warm-up 20 -> PEAK_LR, cosine to 1e-5, 200 epochs, test every 10,
#  teacher momentum 0.9, linear projector 1376. CT batch size comes from the yaml (batch_size per dataset).
#  3D model: patch 4x4x4, window 4x7x7, bf16 autocast, gradient checkpointing.
#
#  START and RESUME: sbatch this file. If you change PEAK_LR, rename --job-name, RUN_NAME and the log paths.
# ======================================================================================================
#SBATCH --job-name=ark3d_swinb224_imnet22k_ricord-luna16_lr0.003_ep200
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_imaging_spring2026
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ztaiyab/ark_runs/ark3d_swinb224_imnet22k_ricord-luna16_lr0.003_ep200/logs/%x_%j.out
#SBATCH --error=/scratch/ztaiyab/ark_runs/ark3d_swinb224_imnet22k_ricord-luna16_lr0.003_ep200/logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -eo pipefail

# ---------------- settings ----------------
PEAK_LR=0.003
RUN_NAME=ark3d_swinb224_imnet22k_ricord-luna16_lr0.003_ep200
RUN_DIR=/scratch/ztaiyab/ark_runs/$RUN_NAME
SIF=/scratch/ztaiyab/ark_runs/containers/ark_vindr.sif
CODE_DIR=/scratch/ztaiyab/ark_code/Ark_unified/Ark_Plus/Pretraining          # git worktree, branch switchable-3d
DATA_LISTS=/scratch/ztaiyab/ark_code/Ark/Ark_Plus/dataset                    # 2D list files (used by 2D+3D runs)
YAML=/scratch/ztaiyab/ark_runs/configs/datasets_config_2d3d.yaml             # 2D entries + LUNA16 + RICORD
WEIGHTS=/scratch/ztaiyab/weights/swin_base_patch4_window7_224_22kto1k.pth

export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
RUN="apptainer exec --nv --env WANDB_MODE=disabled --bind /scratch/ztaiyab:/scratch/ztaiyab $SIF"

for p in "$CODE_DIR/main_ark.py" "$CODE_DIR/unified_swin.py" "$CODE_DIR/lr_schedule.py" "$DATA_LISTS" "$YAML" "$WEIGHTS" "$SIF"; do
  [ -e "$p" ] || { echo "ERROR: missing $p"; exit 1; }; done
mkdir -p "$RUN_DIR/logs" && cd "$RUN_DIR"
[ -f datasets_config.yaml ] || cp "$YAML" datasets_config.yaml
ln -sfn "$DATA_LISTS" dataset
ARK_DIR=Models/unified_swin_base_run/Ark_Plus_RICORD_LUNA16/run
ln -sfn "$ARK_DIR/Ark_Plus_RICORD_LUNA16.pth.tar" checkpoint_latest.pth.tar
ln -sfn "$ARK_DIR/train.log" train_log.txt
ln -sfn "Outputs/unified_swin_base_run/Ark_Plus_RICORD_LUNA16_run_results.txt" test_results.txt
ln -sfn "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" slurm_latest.out

{ echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID}  restart ${SLURM_RESTART_COUNT:-0}  host $(hostname)  peak_lr ${PEAK_LR}"
  echo "  git: $(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD) $(git -C "$CODE_DIR" rev-parse --short HEAD)  uncommitted files: $(git -C "$CODE_DIR" status --short | wc -l)"; } >> run_info.txt
$RUN python -c "import sys, torch, timm; print('env: python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| timm', timm.__version__, '| torch from', torch.__file__, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" | tee -a run_info.txt
case "$(tail -n 1 run_info.txt)" in *"/.local/"*) echo "ERROR: torch loads from ~/.local, not the container"; exit 1;; esac
tail -n 2 run_info.txt | head -n 1
if [ -f checkpoint_latest.pth.tar ]; then echo "RESUMING from $(readlink -f checkpoint_latest.pth.tar)"; else echo "FRESH START"; fi

trap 'echo "[$(date "+%F %T")] time limit close: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1
$RUN python "$CODE_DIR/main_ark.py" \
    --data_set RICORD \
    --data_set LUNA16 \
    --model unified_swin_base \
    --init imagenet \
    --pretrained_weights "$WEIGHTS" \
    --opt sgd \
    --lr $PEAK_LR \
    --warmup-epochs 20 \
    --batch_size 200 \
    --pretrain_epochs 200 \
    --test_epoch 10 \
    --momentum_teacher 0.9 \
    --projector_features 1376 \
    --input_size 224 \
    --img_resize 256 \
    --window_depth 4 \
    --patch_depth 4 \
    --grad_checkpoint True \
    --amp True \
    --freeze_pretrained_epochs 0 \
    --exp_name run \
    --resume True &
wait $!
