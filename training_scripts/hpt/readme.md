# Hyperparameter Tuning

This is a guide to perform hyperparameter search using [Weights & Bias](https://wandb.ai/) on the organization GPU cluster (e.g. `anonymized.url`). 
It is a general approach which is used to find the best hyperparameters for different experiments like entity detection or relation extraction.
We also exploit GPU-cluster to run multiple parameter search in parallel.

## Setup

Follow [these steps](../cluster/README.md#setup).

## Usage

1. Start by creating a sweep using [create_sweep.sh](../cluster/create_sweep.sh). This script uses a sweep config
   (eg: [config_adu.yaml](configs/config_adu.yaml)) provided as argument. Example:

```bash
srun -K --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 --partition RTXA6000-SLT \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER/.cache_slurm:/root/.cache,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh \
--container-workdir="`pwd`" \
--export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,WANDB_API_KEY=YOUR_API_KEY,WANDB_PROJECT=test,WANDB_ENTITY=sam,PYTHONPATH=$PYTHONPATH:`pwd`" \
training_scripts/cluster/create_sweep.sh training_scripts/hpt/configs/config_adu.yaml
```
The `WANDB_API_KEY` can be obtained from [here](https://wandb.ai/authorize) when logged in. 
It will create a sweep for given `WANDB_PROJECT` and `WANDB_ENTITY` using the config passed as parameter to the script. This command will output `SWEEP_ID` on console.
We can also verify the created sweep at https://wandb.ai/WANDB_ENTITY/WANDB_PROJECT/sweeps/SWEEP_ID 

2. Once the sweep is successfully created we can start agent(s) by executing `batch.sh` as following:
```bash
sbatch --export=NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,WANDB_PROJECT='test',\
WANDB_ENTITY='sam',WANDB_NOTES=SWEEP_ID,WANDB_API_KEY=WANDB_API_KEY,PYTHONPATH=$PYTHONPATH:`pwd` \
--job-name=test --array 1-20%4 training_scripts/cluster/batch.sh
```
Replace `SWEEP_ID` by the id obtained after executing step 1. 
Firstly, this command will create an environment by installing required packages for the training using [wrapper.sh](../cluster/wrapper.sh). 
Once setup is finished it initiates multiple agents (mentioned as --array 1-20%4) in parallel.
In above example it runs 20 agents for given sweep with 4 agents running in parallel at a given time. 
Each agent uses the training script mentioned in the sweep config (`program` parameter). We use [train-cross-validation.py](../cross_validation/train-cross-validation.py)
as training script which calls [execution.py](../sweeps/execution.py) where all sweep parameters are handled and training is started.

We can check the cluster job progress [here](http://anonymized.url:3000/d/slurm-current-jobs/current-jobs) under job-name provided in command above (requires to be in the organization VPN).
In order to cancel running jobs, use the `scancel` command. Array jobs are suffixed using numbers , e.g. ID_1, ID_2, so on. Use `scancel ID_1` to cancel a single job or `scancel ID_[1-20]` to cancel all the jobs at once.
Be careful while cancelling jobs, verify your job ids by executing command `squeue -u USERNAME`.

## Debugging

To start an interactive session, e.g. to check out what is in the image, execute something like this:
```
srun -K --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 --time=01:00:00 -p RTX2080Ti-SLT \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER/.cache_slurm:/root/.cache,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh \
--container-workdir="`pwd`" \
--export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
--pty bash -i
```
Note the `--time` parameter that kills your session automatically if you accidentally forgot it.
