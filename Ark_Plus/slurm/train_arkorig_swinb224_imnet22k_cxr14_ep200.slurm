#!/bin/bash
# ======================================================================================================
#  Original Ark+ pretraining | Swin-B (window 7, 224 px) | ImageNet-22k->1k init | linear projector 1376
#  Datasets : ChestX-ray14
#  Recipe   : Ark+ README "Ark+6 Swin Base" command, unchanged: SGD, lr 0.3, warm-up 20, batch 200,
#             200 epochs, test every 10, teacher momentum 0.9, --projector_features 1376 (no --use_mlp),
#             input 224 / resize 256 / 8 workers / ema_mode epoch (README defaults, written out here).
#
#  START and RESUME: sbatch this file. It always passes --resume True; Ark starts fresh if no checkpoint
#  exists, otherwise continues after the last completed epoch. Rename --job-name, RUN_NAME and the two
#  log paths together if you change the setup, so a different setup never resumes into this folder.
#
#  Fixed learning-rate schedule instead: set CODE_DIR=.../Pretraining_original_lrfix and rename the run
#  to arkorig-lrfix_... (--epochs already defaults to 200, so the cosine schedule spans the run).
# ======================================================================================================
#SBATCH --job-name=arkorig_swinb224_imnet22k_cxr14_ep200
#SBATCH --partition=public
##SBATCH --qos=class                         # remove one '#' (and set --account) to use the class QOS
##SBATCH --account=class_imaging_spring2026
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                           # 64G was too little for Ark's ten-crop test pass
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ztaiyab/ark_runs/arkorig_swinb224_imnet22k_cxr14_ep200/logs/%x_%j.out
#SBATCH --error=/scratch/ztaiyab/ark_runs/arkorig_swinb224_imnet22k_cxr14_ep200/logs/%x_%j.err
#SBATCH --open-mode=append
# Automatic continuation at the time limit: SLURM signals 15 min before the end, the job requeues itself
# and resumes. To turn it off, delete the next two lines and the 'trap' line further down.
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -eo pipefail

# ---------------- paths ----------------
RUN_NAME=arkorig_swinb224_imnet22k_cxr14_ep200
RUN_DIR=/scratch/ztaiyab/ark_runs/$RUN_NAME
SIF=/scratch/ztaiyab/ark_runs/containers/ark_vindr.sif                    # torch 1.13.1 | CUDA 11.6 | timm 0.5.4
CODE_DIR=/scratch/ztaiyab/ark_code/Ark/Ark_Plus/Pretraining_original       # original Ark+ (+ the 2 resume-only fixes)
WEIGHTS=/scratch/ztaiyab/weights/swin_base_patch4_window7_224_22kto1k.pth  # = the README URL, downloaded once

# ---------------- environment (inside the container) ----------------
export WANDB_MODE=disabled          # original Ark calls wandb.log without wandb.init
export PYTHONNOUSERSITE=1           # never use packages from ~/.local
RUN="apptainer exec --nv --env WANDB_MODE=disabled --bind /scratch/ztaiyab:/scratch/ztaiyab $SIF"

# ---------------- run folder: Ark writes Models/ and Outputs/ here ----------------
mkdir -p "$RUN_DIR/logs" && cd "$RUN_DIR"
[ -f datasets_config.yaml ] || cp "$CODE_DIR/datasets_config.yaml" .   # snapshot at the first start
if   [ -d "$CODE_DIR/dataset" ];    then ln -sfn "$(cd "$CODE_DIR/dataset" && pwd -P)" dataset
elif [ -d "$CODE_DIR/../dataset" ]; then ln -sfn "$(cd "$CODE_DIR/../dataset" && pwd -P)" dataset
else echo "ERROR: no dataset/ folder next to $CODE_DIR"; exit 1; fi

# shortcuts to Ark's deep paths (Models/<model>_<exp_name>/Ark_Plus_<datasets>/<exp_name>/...)
ARK_DIR=Models/swin_base_run/Ark_Plus_ChestXray14/run
ln -sfn "$ARK_DIR/Ark_Plus_ChestXray14.pth.tar" checkpoint_latest.pth.tar
ln -sfn "$ARK_DIR/train.log" train_log.txt
ln -sfn "Outputs/swin_base_run/Ark_Plus_ChestXray14_run_results.txt" test_results.txt
ln -sfn "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" slurm_latest.out

# ---------------- record this start: environment, code checksums ----------------
{ echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID}  restart ${SLURM_RESTART_COUNT:-0}  host $(hostname)"
  (cd "$CODE_DIR" && md5sum main_ark.py engine.py models.py trainer.py dataloader.py | sed 's/^/  /'); } >> run_info.txt
$RUN python -c "import sys, torch, timm; print('env: python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| timm', timm.__version__, '| torch from', torch.__file__, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" | tee -a run_info.txt
case "$(tail -n 1 run_info.txt)" in *"/.local/"*) echo "ERROR: torch loads from ~/.local, not the container"; exit 1;; esac
if [ -f checkpoint_latest.pth.tar ]; then echo "RESUMING from $(readlink -f checkpoint_latest.pth.tar)"; else echo "FRESH START"; fi

# ---------------- train ----------------
trap 'echo "[$(date "+%F %T")] time limit close: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1
$RUN python "$CODE_DIR/main_ark.py" \
    --data_set ChestXray14 \
    --model swin_base \
    --init imagenet \
    --pretrained_weights "$WEIGHTS" \
    --opt sgd \
    --lr 0.3 \
    --warmup-epochs 20 \
    --batch_size 200 \
    --pretrain_epochs 200 \
    --test_epoch 10 \
    --momentum_teacher 0.9 \
    --projector_features 1376 \
    --input_size 224 \
    --img_resize 256 \
    --exp_name run \
    --resume True &
wait $!
