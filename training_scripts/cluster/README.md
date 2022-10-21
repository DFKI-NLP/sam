# GPU Cluster

This directory contains some scripts to support training execution on a SLURM GPU cluster.

## Setup

One time setup at the GPU cluster:
1. Clone the project and switch into directory: `git clone git@github.com:ANONYMIZED_AUTHOR/SAM.git && cd SAM`
2. Change permission for setup scripts:
   1. `chmod +x training_scripts/cluster/wrapper.sh`
   2. `chmod +x training_scripts/cluster/create_sweep.sh`
   3. (Optional) `chmod +x training_scripts/cluster/torch_configs/torch_versions_rtxa6000.sh`
3. Create a mount point on the host for the cache (you can choose any directory, just modify the mapping in the next steps accordingly): `mkdir ~/.cache_slurm`
4. (Optional) Verify setup by successfully executing a command, e.g. execute the following (note the mount from the host cache directory you created above to the cache directory in the 
image: `/home/$USER/.cache_slurm:/root/.cache`, the host directory has to exist and you have to have write access!):
```bash
srun -K --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 -p RTX2080Ti-SLT \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER/.cache_slurm:/root/.cache,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh \
--container-workdir="`pwd`" \
--export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
training_scripts/cluster/wrapper.sh pip list
```
NOTE: `wrapper.sh` installs current stable version of pytorch which works well with RTX2080Ti. Both pytorch and cuda version can be 
found in `requirements.txt`. If your GPU do not support these pytorch and cuda version, then
you can create a config in `training_scripts/cluster/torch_configs` folder and pass
this config as environment variable for $TORCH_VERSION_FILE while running experiments.

For example, RTX-A6000 do not support the versions specified in requirements.txt, so we have
a config `torch_versions_rtxa6000.sh` in `training_scripts/cluster/torch_configs` which specifies
required working versions of torch and cuda for RTX-A6000. Then we run experiments as

```bash
srun -K --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 -p RTXA6000-SLT \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER/.cache_slurm:/root/.cache,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh \
--container-workdir="`pwd`" \
--export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,TORCH_VERSION_FILE=training_scripts/cluster/torch_configs/torch_versions_rtxa6000.sh" \
training_scripts/cluster/wrapper.sh pip list
```

With $TORCH_VERSION_FILE value set, wrapper.sh reinstall torch with specific version
for GPU as defined in $TORCH_VERSION_FILE (allennlp has enforced a lower torch
version which would create conflict when defined in requirements.txt)
