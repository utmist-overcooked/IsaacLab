#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --job-name=cobot-sanity
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/zachtang/logs/%x-%j.out
# Runs the comm-vs-no-comm sanity comparison plus a flip-knob check.

set -eo pipefail
module --force purge && module load StdEnv/2023 python/3.12.4
unset PYTHONPATH PIP_CONFIG_FILE
cd /project/aip-mtaylor3/zachtang/IsaacLab
source env_isaaclab/bin/activate
export LD_LIBRARY_PATH=$PWD/.cuda-shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMNI_KIT_ACCEPT_EULA=YES XDG_CACHE_HOME=$SCRATCH/.cache HF_HOME=$SCRATCH/hf
mkdir -p $SCRATCH/ovcache/{cache_ov,share_ov,nvidia,nv,omnilogs} $SCRATCH/logs

echo "########## channel (comm on) ##########"
./isaaclab.sh -p scripts/cobot/run_sanity.py --headless --num_envs 32 --steps 250 --comm_mode channel
echo "########## none (comm off) ##########"
./isaaclab.sh -p scripts/cobot/run_sanity.py --headless --num_envs 32 --steps 250 --comm_mode none
echo "########## channel + flips (knob 1) ##########"
./isaaclab.sh -p scripts/cobot/run_sanity.py --headless --num_envs 32 --steps 250 --comm_mode channel --flip_prob 0.05
echo "########## none + flips ##########"
./isaaclab.sh -p scripts/cobot/run_sanity.py --headless --num_envs 32 --steps 250 --comm_mode none --flip_prob 0.05
