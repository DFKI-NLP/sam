#!/bin/bash

if [[ $SLURM_LOCALID == 0 ]]; then

  # path has to start from working directory root
  echo "install requirements from: requirements.txt"
  pip install -r requirements.txt
  if [[ $TORCH_VERSION_FILE ]]; then
    echo "load torch versions from file: $TORCH_VERSION_FILE"

    source $TORCH_VERSION_FILE

    echo "update to the following versions:"
    echo "  PYTORCH_CUDA_VERSION: $PYTORCH_CUDA_VERSION"
    echo "  PYTORCH_VERSION: $PYTORCH_VERSION"
    echo "  TORCHAUDIO_VERSION: $TORCHAUDIO_VERSION"
    echo "  TORCHVISION_VERSION: $TORCHVISION_VERSION"

    pip uninstall --yes torch
    pip uninstall --yes torchvision
    pip uninstall --yes torchtext
    pip3 install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    -f https://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION}/torch_stable.html
  fi
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# This runs your wrapped command
"$@"
