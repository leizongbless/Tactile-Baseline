# #!/bin/bash
# 5hour = 3600
echo "sleep for 18000s"
sleep 18000

GPU_ID=2

horizon=96

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="/home/kywang/projects/efficient_robot_sys/data/ckpts/timer_no_lift_higher_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=timer_no_lift_higher_rdp_ldp_60hz_horizon${horizon}
SEARCH_PATH="./data/outputs"



# AT_LOAD_DIR="/home/kywang/projects/efficient_robot_sys/data/ckpts/timer_no_lift_higher_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon96/latest.ckpt"
AT_LOAD_DIR="/home/kywang/projects/Tactile-Baseline/data/outputs/2026.03.23/11.35.54_train_vae_real_wipe_timer_no_lift_higher_rdp_vae_60hz_horizon96/checkpoints/latest.ckpt"

# # # Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    +policy.change_kernel_size=False \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps_without_tactile \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_ldp_${TIMESTAMP} \
    at=at_wipe_lift \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False \
    at.policy.n_embed=10
