# #!/bin/bash

GPU_ID=2
horizon=12
num_epochs=400
batch_size=256
suite=libero_object

TASK_NAME="wipe"
DATASET_PATH="/home/kywang/projects/efficient_robot_sys/data/ckpts/${suite}_wo_img/rdp_zarr"
LOGGING_MODE="online"
TIMESTAMP=${suite}_vae_horizon${horizon}_num_epochs${num_epochs}
SEARCH_PATH="./data/outputs"

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training Asymmetric Tokenizer..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_at_workspace \
    task=at_libero \
    task.dataset_path=${DATASET_PATH} \
    task.dataset.relative_action=False \
    task.name=real_${TASK_NAME}_${TIMESTAMP} \
    at=at_wipe_lift \
    logging.mode=${LOGGING_MODE} \
    at.dataset_obs_temporal_downsample_ratio=1 \
    at.horizon=${horizon} \
    at.n_obs_steps=1 \
    at.policy.use_rnn_decoder=False \
    at.policy.n_embed=7 \
    dataloader.batch_size=${batch_size} \
    training.num_epochs=${num_epochs}