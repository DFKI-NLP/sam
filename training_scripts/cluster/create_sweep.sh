#!/bin/bash

function creat_sweep()
{
python - <<START
import wandb
import sys
import yaml
with open("$1") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
sweep_id = wandb.sweep(config_dict)
print (sweep_id)
START
}

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then

  # path has to start from working directory root
  #XDG_CACHE_HOME="/home/$USER/.cache/pip" pip install -r training_scripts/cluster/requirements.txt
  echo "install wandb to create sweep"
  pip install wandb==0.10.33
  OUTPUT=$(creat_sweep "$1" | cut -d$'\n' -f 3)
  echo SWEEP ID: "$OUTPUT"
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi