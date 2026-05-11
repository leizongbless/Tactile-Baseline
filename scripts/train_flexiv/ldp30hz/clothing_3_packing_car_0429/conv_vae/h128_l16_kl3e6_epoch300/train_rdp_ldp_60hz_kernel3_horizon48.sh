#!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-0}
horizon=${horizon:-48}
num_epochs=${num_epochs:-400}

TASK_NAME="wipe"
DATASET_PATH=${DATASET_PATH:-"/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/flexiv_clothing_3_packing_0429_30hz/rdp_zarr"}
LOGGING_MODE=${LOGGING_MODE:-"online"}

VAE_CONFIG_NAME="h128_l16_kl3e6_epoch300"
VAE_CKPT_ROOT=${VAE_CKPT_ROOT:-"/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/flexiv_clothing_3_packing_0429_30hz/ckpts_abs/conv_vae/n_embed_20"}
AT_LOAD_DIR=${AT_LOAD_DIR:-"${VAE_CKPT_ROOT}/${VAE_CONFIG_NAME}/latest.ckpt"}

hidden_dim=128
n_latent_dims=16
n_embed=20
downsample_factor=2
blocks_per_level=3
vae_kernel_size=5
kl_multiplier=3e-6
policy_kernel_size=3

TIMESTAMP=${TIMESTAMP:-"cloth0429_ldp_tconv_${VAE_CONFIG_NAME}_h${horizon}"}
LOGGING_ID=${LOGGING_ID:-"ldp_tconv_${VAE_CONFIG_NAME}_h${horizon}"}

echo ""
echo "Stage 2: training Latent Diffusion Policy with TemporalConvVAE ${VAE_CONFIG_NAME}..."
echo "AT_LOAD_DIR=${AT_LOAD_DIR}"
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_temporal_conv_vae_real_image_workspace \
    policy.kernel_size=${policy_kernel_size} \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps_without_tactile_dual_arm \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=${TIMESTAMP} \
    at=temporal_conv_vae_wipe_lift \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.n_obs_steps=1 \
    at.policy.hidden_dim=${hidden_dim} \
    at.policy.n_latent_dims=${n_latent_dims} \
    at.policy.n_embed=${n_embed} \
    at.policy.downsample_factor=${downsample_factor} \
    at.policy.blocks_per_level=${blocks_per_level} \
    at.policy.kernel_size=${vae_kernel_size} \
    at.policy.kl_multiplier=${kl_multiplier} \
    at.policy.use_rnn_decoder=False \
    task.env_runner.use_latent_action_with_rnn_decoder=False \
    training.num_epochs=${num_epochs} \
    logging.id=${LOGGING_ID}
