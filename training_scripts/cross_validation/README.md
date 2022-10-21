# Cross Validation Training

In order to train model with k-fold cross validation we use [train-cross-validation.py](train-cross-validation.py) script.
We can train model in GPU-cluster as well as in local GPU. 

## Usage

### 1. Cluster execution

1. Setup at the GPU cluster, see [here](../cluster/README.md#setup).
2. Now run following command to start training

```bash
srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-task=2 -p RTXA6000-SLT \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER/.cache_slurm:/root/.cache,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh \
--container-workdir="`pwd`" \
--export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,WANDB_API_KEY=YOUR_API_KEY,WANDB_PROJECT=test,WANDB_ENTITY=sam,PYTHONPATH=$PYTHONPATH:`pwd`" \
training_scripts/cluster/wrapper.sh \
python training_scripts/cross_validation/train-cross-validation.py \
--num_folds 5 \
-s experiments/training/adu/adu_cv \
train \
-f allennlp_configs/adu_best.jsonnet \
-o "{data_loader:{batch_size:16}}"
```

### 2. Local GPU execution

1. Clone project and switch into directory: `git clone git@github.com:ANONYMIZED_AUTHOR/SAM.git && cd SAM`
2. Run following command to train with cross validation
```bash
CUDA_VISIBLE_DEVICES=7,WANDB_API_KEY=YOUR_API_KEY,WANDB_PROJECT=test,WANDB_ENTITY=sam" \
python training_scripts/cross_validation/train-cross-validation.py \
--num_folds 5 \
-s experiments/training/adu/adu_cv \
train \
-f allennlp_configs/adu_best.jsonnet \
-o "{data_loader:{batch_size:16}}"
```
**NOTE**: 
1. Put available CUDA device for environment variable `CUDA_VISIBLE_DEVICES`.
2. Ensure API key is set for env variable `WANDB_API_KEY` (other two variables are optional: `WANDB_PROJECT`, `WANDB_ENTITY`).