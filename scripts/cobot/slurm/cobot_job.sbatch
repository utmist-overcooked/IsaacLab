#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --job-name=cobot
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/zachtang/logs/%x-%j.out
# Usage: sbatch cobot_job.sbatch <script.py> [args...]
# Runs one IsaacLab python script with the Vulcan environment preamble.

set -eo pipefail
module --force purge && module load StdEnv/2023 python/3.12.4
unset PYTHONPATH PIP_CONFIG_FILE
cd /project/aip-mtaylor3/zachtang/IsaacLab
source env_isaaclab/bin/activate
export LD_LIBRARY_PATH=$PWD/.cuda-shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMNI_KIT_ACCEPT_EULA=YES XDG_CACHE_HOME=$SCRATCH/.cache HF_HOME=$SCRATCH/hf
mkdir -p $SCRATCH/ovcache/{cache_ov,share_ov,nvidia,nv,omnilogs} $SCRATCH/logs

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# background GPU utilization sampler
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 10 \
    > $SCRATCH/logs/gpuutil-$SLURM_JOB_ID.csv &
SAMPLER_PID=$!

./isaaclab.sh -p "$@"
kill $SAMPLER_PID || true
