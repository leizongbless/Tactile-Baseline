#!/bin/bash#!/bin/bash
python infer_forcemimic.py \
  --config-name train_diffusion_unet_real_image_workspace \
  task=forcemimic \
  task.name=forcemimic_infer \
  logging.mode=offline \
  load_pca_path=/home/robotics/Prometheus/reactive_diffusion_policy/tactile_pca/board \
  load_ckpt_path=/home/robotics/Tactile-Baseline/ckpt/latest_dp_bb.ckpt