#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_forcemimic_image_workspace \
    task=forcemimic \
    task.dataset_path=/home/tare/wks/dataset/tactile/peel/peel_hg/rdp_zarr \
    task.name=dp_forcemimic_peel_hg \
    logging.mode=offline







