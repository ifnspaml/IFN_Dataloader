#!/bin/bash

export IFN_DIR_DATASET=/home/simon/IFN-Cluster/beegfs/work/shared
export IFN_DIR_CHECKPOINT=/home/simon/IFN_workspace/Checkpoints

python3 cityscapes_dataloader_example.py


echo "Completed job on "$(hostname)
