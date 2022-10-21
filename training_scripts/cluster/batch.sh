#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array 1-20%4
#SBATCH --job-name test
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-cpu=24G
#SBATCH --partition RTXA6000

srun -K --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER/.cache_slurm:/root/.cache,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh \
--container-workdir="`pwd`" \
training_scripts/cluster/wrapper.sh \
python training_scripts/sweeps/start_agent.py -c 1