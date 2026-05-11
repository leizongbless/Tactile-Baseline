#!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-1}
horizon=${horizon:-48}
num_epochs=${num_epochs:-300}

TASK_NAME="wipe"
DATASET_PATH=${DATASET_PATH:-"/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/flexiv_clothing_3_packing_0429_30hz/rdp_zarr"}
LOGGING_MODE=${LOGGING_MODE:-"online"}
TIMESTAMP=${TIMESTAMP:-"cloth0429_temporal_conv_vae_h128_l24_kl3e6_h${horizon}"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

num_workers=${num_workers:-32}
batch_size=${batch_size:-128}
sample_every=${sample_every:-50}

echo "Training TemporalConvVAE h128_l24_kl3e6 on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_temporal_conv_vae_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_at_24fps_dual_arm \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=${TIMESTAMP} \
    logging.mode=${LOGGING_MODE} \
    logging.project=at \
    at=temporal_conv_vae_wipe_lift \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.n_obs_steps=1 \
    at.policy.hidden_dim=128 \
    at.policy.n_latent_dims=24 \
    at.policy.n_embed=20 \
    at.policy.downsample_factor=2 \
    at.policy.blocks_per_level=3 \
    at.policy.kernel_size=5 \
    at.policy.kl_multiplier=3e-6 \
    training.num_epochs=${num_epochs} \
    training.sample_every=${sample_every} \
    training.resume=False \
    training.checkpoint_every=1 \
    dataloader.num_workers=${num_workers} \
    dataloader.batch_size=${batch_size}
