# #!/bin/bash

set -euo pipefail

ROOT_DIR="/mnt/data/kywang/4090_env/projects/efficient_robot_sys/third_party/Tactile-Baseline"
TRAIN_SCRIPT="scripts/train_flexiv/rdp_vae30hz/clothing_3_packing_0429_30hz/train_temporal_conv_vae_horizon48.sh"
num_epochs=${num_epochs:-100}
LOGGING_MODE=${LOGGING_MODE:-online}
num_workers=${num_workers:-16}
batch_size=${batch_size:-128}

launch_tmux_run() {
    local session="$1"
    local gpu_id="$2"
    local model_type="$3"
    shift 3

    echo "Launching ${model_type} in tmux session ${session} on GPU ${gpu_id}"
    tmux new-session -d -s "${session}" \
        "cd ${ROOT_DIR} && \
         source /home/kywang/miniconda3/etc/profile.d/conda.sh && \
         conda activate rdp && \
         GPU_ID=${gpu_id} \
         MODEL_TYPE=${model_type} \
         RUN_NAME=cloth0429_${model_type}_policy_latent_e${num_epochs} \
         LOGGING_MODE=${LOGGING_MODE} \
         num_epochs=${num_epochs} \
         num_workers=${num_workers} \
         batch_size=${batch_size} \
         N_EMBED=20 \
         DOWNSAMPLE_FACTOR=2 \
         $* \
         bash ${TRAIN_SCRIPT}"
}

launch_tmux_run tcvae_policy_base_e100 2 policy_base_h128_l24_kl1e6 \
    HIDDEN_DIM=128 N_LATENT_DIMS=24 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=1e-6

launch_tmux_run tcvae_policy_kl3e6_e100 0 policy_h128_l24_kl3e6 \
    HIDDEN_DIM=128 N_LATENT_DIMS=24 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=3e-6

launch_tmux_run tcvae_policy_kl1e5_e100 2 policy_h128_l24_kl1e5 \
    HIDDEN_DIM=128 N_LATENT_DIMS=24 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=1e-5

launch_tmux_run tcvae_policy_h96_kl3e6_e100 0 policy_h96_l24_kl3e6 \
    HIDDEN_DIM=96 N_LATENT_DIMS=24 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=3e-6

launch_tmux_run tcvae_policy_l16_kl3e6_e100 2 policy_h128_l16_kl3e6 \
    HIDDEN_DIM=128 N_LATENT_DIMS=16 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=3e-6

launch_tmux_run tcvae_policy_h160_kl3e6_e100 0 policy_h160_l24_kl3e6 \
    HIDDEN_DIM=160 N_LATENT_DIMS=24 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=3e-6
