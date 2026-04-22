# #!/bin/bash
# 5hour = 3600
# echo "sleep for 14400"
# sleep 14400

GPU_ID=1

horizon=96

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/press_car_0418_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=hit_mouse_merge_ldp_horizon${horizon}
SEARCH_PATH="./data/outputs"
num_epochs=450 
kernel_size=3


AT_LOAD_DIR="/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/press_car_0418_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon${horizon}/latest.ckpt"

# # # Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    policy.kernel_size=${kernel_size} \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps_without_tactile \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_ldp_kernel${kernel_size}_${TIMESTAMP} \
    at=at_wipe_lift \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False \
    at.policy.n_embed=10 \
    training.num_epochs=${num_epochs}
