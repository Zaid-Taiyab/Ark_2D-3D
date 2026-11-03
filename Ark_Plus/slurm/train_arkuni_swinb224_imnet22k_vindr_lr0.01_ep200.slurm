#!/bin/bash
# ======================================================================================================
#  2D+3D | unified switchable Swin-B (branch switchable-3d) | ImageNet-22k->1k (inflated for CT)
#  Datasets : VinDr-CXR only. Control for the 2D+3D runs: same unified code, lr 0.01, bf16, no CT
#  No DRR heads (drr_head/return_drr false in the yaml): the two modalities share only the encoder.
#  Learning rates: X-ray steps PEAK_LR = 0.01 (LR_2D, from the lr range test); CT steps PEAK_LR x LR_SCALE_3D
#  = 0.0003 (LR_3D, range test + sweep). Same warm-up 20 -> cosine over 200 epochs for both.
#  Compare with: RICORD alone at 0.0003 (ark3d_..._ricord_lr0.0003) and VinDr alone at 0.01.
#  START and RESUME: sbatch this file.
# ======================================================================================================
#SBATCH --job-name=arkuni_swinb224_imnet22k_vindr_lr0.01_ep200
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_imaging_spring2026
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ztaiyab/ark_runs/arkuni_swinb224_imnet22k_vindr_lr0.01_ep200/logs/%x_%j.out
#SBATCH --error=/scratch/ztaiyab/ark_runs/arkuni_swinb224_imnet22k_vindr_lr0.01_ep200/logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -eo pipefail

# ---------------- settings ----------------
PEAK_LR=0.01                                                                # X-ray steps (LR_2D)
LR_SCALE_3D=0.03                                                            # CT steps: 0.01 x 0.03 = 0.0003 (LR_3D)
RUN_NAME=arkuni_swinb224_imnet22k_vindr_lr0.01_ep200
RUN_DIR=/scratch/ztaiyab/ark_runs/$RUN_NAME
SIF=/scratch/ztaiyab/ark_runs/containers/ark_vindr.sif
CODE_DIR=/scratch/ztaiyab/ark_code/Ark_unified/Ark_Plus/Pretraining          # git worktree, branch switchable-3d
DATA_LISTS=/scratch/ztaiyab/ark_code/Ark/Ark_Plus/dataset                    # 2D list files (used by 2D+3D runs)
YAML=/scratch/ztaiyab/ark_runs/configs/datasets_config_2d3d.yaml             # 2D entries + LUNA16 + RICORD
WEIGHTS=/scratch/ztaiyab/weights/swin_base_patch4_window7_224_22kto1k.pth

export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
RUN="apptainer exec --nv --env WANDB_MODE=disabled --bind /scratch/ztaiyab:/scratch/ztaiyab $SIF"

grep -q interleave "$CODE_DIR/main_ark.py" || { echo "ERROR: main_ark.py lacks --interleave (apply patch_interleave.py)"; exit 1; }
grep -q clip_grad_3d "$CODE_DIR/main_ark.py" || { echo "ERROR: main_ark.py lacks --clip_grad_3d (apply patch_ct_clip.py)"; exit 1; }
grep -q optimizer_3d "$CODE_DIR/engine.py" || { echo "ERROR: engine.py lacks the modality-optimizer patch"; exit 1; }
grep -q lr_scale_3d "$CODE_DIR/main_ark.py" || { echo "ERROR: $CODE_DIR has no --lr_scale_3d (apply patch_modality_lr.py)"; exit 1; }
for p in "$CODE_DIR/main_ark.py" "$CODE_DIR/unified_swin.py" "$CODE_DIR/lr_schedule.py" "$DATA_LISTS" "$YAML" "$WEIGHTS" "$SIF"; do
  [ -e "$p" ] || { echo "ERROR: missing $p"; exit 1; }; done
mkdir -p "$RUN_DIR/logs" && cd "$RUN_DIR"
[ -f datasets_config.yaml ] || cp "$YAML" datasets_config.yaml
ln -sfn "$DATA_LISTS" dataset
ARK_DIR=Models/unified_swin_base_run/Ark_Plus_VinDrCXR/run
ln -sfn "$ARK_DIR/Ark_Plus_VinDrCXR.pth.tar" checkpoint_latest.pth.tar
ln -sfn "$ARK_DIR/train.log" train_log.txt
ln -sfn "Outputs/unified_swin_base_run/Ark_Plus_VinDrCXR_run_results.txt" test_results.txt
ln -sfn "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" slurm_latest.out

{ echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID}  restart ${SLURM_RESTART_COUNT:-0}  host $(hostname)  peak_lr ${PEAK_LR} (CT x${LR_SCALE_3D})"
  echo "  git: $(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD) $(git -C "$CODE_DIR" rev-parse --short HEAD)  uncommitted files: $(git -C "$CODE_DIR" status --short | wc -l)"; } >> run_info.txt
$RUN python -c "import sys, torch, timm; print('env: python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| timm', timm.__version__, '| torch from', torch.__file__, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" | tee -a run_info.txt
case "$(tail -n 1 run_info.txt)" in *"/.local/"*) echo "ERROR: torch loads from ~/.local, not the container"; exit 1;; esac
tail -n 2 run_info.txt | head -n 1
if [ -f checkpoint_latest.pth.tar ]; then echo "RESUMING from $(readlink -f checkpoint_latest.pth.tar)"; else echo "FRESH START"; fi

trap 'echo "[$(date "+%F %T")] time limit close: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1
$RUN python "$CODE_DIR/main_ark.py" \
    --data_set VinDrCXR \
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
