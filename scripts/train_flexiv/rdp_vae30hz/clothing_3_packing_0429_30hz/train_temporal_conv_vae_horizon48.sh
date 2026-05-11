# #!/bin/bash

set -euo pipefail

GPU_ID=${GPU_ID:-0}
MODEL_TYPE=${MODEL_TYPE:-tcvae_m}
horizon=${horizon:-48}
num_epochs=${num_epochs:-100}

TASK_NAME="wipe"
DATASET_PATH=${DATASET_PATH:-"/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/flexiv_clothing_3_packing_0429_30hz/rdp_zarr"}
LOGGING_MODE=${LOGGING_MODE:-"online"}
RUN_NAME=${RUN_NAME:-"cloth0429_${MODEL_TYPE}_h${horizon}_eval50"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

num_workers=${num_workers:-16}
batch_size=${batch_size:-128}
sample_every=${sample_every:-50}

echo "Training ${MODEL_TYPE} on GPU ${GPU_ID}"

COMMON_ARGS=(
    "task=real_${TASK_NAME}_image_gelsight_emb_at_24fps_dual_arm"
    "task.dataset_path=${DATASET_PATH}"
    "task.dataset.relative_action=False"
    "task.name=${RUN_NAME}"
    "logging.mode=${LOGGING_MODE}"
    "logging.project=at"
    "training.num_epochs=${num_epochs}"
    "training.sample_every=${sample_every}"
    "training.resume=False"
    "training.checkpoint_every=1"
    "dataloader.num_workers=${num_workers}"
    "dataloader.batch_size=${batch_size}"
)

if [[ "${MODEL_TYPE}" == "baseline_wide" ]]; then
    N_LATENT_DIMS=${N_LATENT_DIMS:-16}
    CONV_LATENT_DIMS=${CONV_LATENT_DIMS:-64}
    CONV_LAYER_NUM=${CONV_LAYER_NUM:-2}
    N_EMBED=${N_EMBED:-32}
    MLP_LAYER_NUM=${MLP_LAYER_NUM:-2}
    KL_MULTIPLIER=${KL_MULTIPLIER:-1e-6}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        --config-name=train_at_workspace \
        "${COMMON_ARGS[@]}" \
        at=at_wipe_lift \
        at.dataset_obs_temporal_downsample_ratio=1 \
        at.horizon=${horizon} \
        at.n_obs_steps=1 \
        at.policy.use_conv_encoder=True \
        at.policy.use_rnn_decoder=False \
        at.policy.n_latent_dims=${N_LATENT_DIMS} \
        at.policy.conv_latent_dims=${CONV_LATENT_DIMS} \
        at.policy.conv_layer_num=${CONV_LAYER_NUM} \
        at.policy.n_embed=${N_EMBED} \
        at.policy.mlp_layer_num=${MLP_LAYER_NUM} \
        at.policy.kl_multiplier=${KL_MULTIPLIER}
else
    HIDDEN_DIM=${HIDDEN_DIM:-128}
    N_LATENT_DIMS=${N_LATENT_DIMS:-32}
    N_EMBED=${N_EMBED:-32}
    DOWNSAMPLE_FACTOR=${DOWNSAMPLE_FACTOR:-4}
    BLOCKS_PER_LEVEL=${BLOCKS_PER_LEVEL:-3}
    KERNEL_SIZE=${KERNEL_SIZE:-5}
    KL_MULTIPLIER=${KL_MULTIPLIER:-1e-6}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        --config-name=train_temporal_conv_vae_workspace \
        "${COMMON_ARGS[@]}" \
        at=temporal_conv_vae_wipe_lift \
        at.dataset_obs_temporal_downsample_ratio=1 \
        at.horizon=${horizon} \
        at.n_obs_steps=1 \
        at.policy.hidden_dim=${HIDDEN_DIM} \
        at.policy.n_latent_dims=${N_LATENT_DIMS} \
        at.policy.n_embed=${N_EMBED} \
        at.policy.downsample_factor=${DOWNSAMPLE_FACTOR} \
        at.policy.blocks_per_level=${BLOCKS_PER_LEVEL} \
        at.policy.kernel_size=${KERNEL_SIZE} \
        at.policy.kl_multiplier=${KL_MULTIPLIER}
fi
