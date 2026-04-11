# #!/bin/bash

GPU_ID=0
horizon=96

TASK_NAME="wipe"
# Point to the dataset directory that contains 'replay_buffer.zarr'
# DATASET_PATH="/data/kywang/projects/tactile_il/data/processed/vase_new_C/rdp_zarr"
DATASET_PATH="/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/press_car_0407_60hz/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=press_car_0407_rdp_vae_60hz_horizon${horizon}
SEARCH_PATH="./data/outputs"
num_epochs=600

# optimize dataload
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
num_workers=64
batch_size=256

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training Asymmetric Tokenizer..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_at_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_at_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=${TIMESTAMP} \
    at=at_wipe_lift \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False \
    at.policy.n_embed=10 \
    training.num_epochs=${num_epochs} \
    dataloader.num_workers=${num_workers} \
    dataloader.batch_size=${batch_size}