#!/bin/bash#!/bin/bash
python infer_kinedex.py \
  --config-name train_kinedex_image_workspace \
  task=kinedex \
  task.name=kinedex_infer \
  logging.mode=offline \
  load_pca_path=/home/robotics/Prometheus/reactive_diffusion_policy/tactile_pca/board \
  load_ckpt_path=/home/robotics/Tactile-Baseline/ckpt/latest_dp_bb.ckpt