#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_12fps \
    task.dataset_path=/home/tare/wks/dataset/tactile/peel/peel_hg/rdp_zarr \
    task.name=dp_tactile_peel_hg \
    logging.mode=offline