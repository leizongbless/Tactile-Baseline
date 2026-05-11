# #!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_temporal_conv_vae_horizon48.sh"

IFS=',' read -ra GPU_LIST <<< "${GPUS:-0,1,2}"
num_epochs=${num_epochs:-50}
LOGGING_MODE=${LOGGING_MODE:-online}

launch_run() {
    local run_idx="$1"
    local gpu_id="$2"
    local model_type="$3"
    shift 3

    echo "Launching ${model_type} on GPU ${gpu_id}"
    env \
        GPU_ID="${gpu_id}" \
        MODEL_TYPE="${model_type}" \
        RUN_NAME="cloth0429_${model_type}_h48_eval50" \
        num_epochs="${num_epochs}" \
        LOGGING_MODE="${LOGGING_MODE}" \
        "$@" \
        bash "${TRAIN_SCRIPT}" &
}

run_idx=0

launch_run "${run_idx}" "${GPU_LIST[$((run_idx % ${#GPU_LIST[@]}))]}" t24_h96_l24_b3_k5_kl1e6 \
    HIDDEN_DIM=96 N_LATENT_DIMS=24 N_EMBED=20 DOWNSAMPLE_FACTOR=2 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=1e-6
run_idx=$((run_idx + 1))

launch_run "${run_idx}" "${GPU_LIST[$((run_idx % ${#GPU_LIST[@]}))]}" t24_h96_l24_b3_k5_kl1e7 \
    HIDDEN_DIM=96 N_LATENT_DIMS=24 N_EMBED=20 DOWNSAMPLE_FACTOR=2 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=1e-7
run_idx=$((run_idx + 1))

launch_run "${run_idx}" "${GPU_LIST[$((run_idx % ${#GPU_LIST[@]}))]}" t24_h128_l24_b3_k5 \
    HIDDEN_DIM=128 N_LATENT_DIMS=24 N_EMBED=20 DOWNSAMPLE_FACTOR=2 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=1e-6
run_idx=$((run_idx + 1))

launch_run "${run_idx}" "${GPU_LIST[$((run_idx % ${#GPU_LIST[@]}))]}" t24_h96_l32_b3_k5 \
    HIDDEN_DIM=96 N_LATENT_DIMS=32 N_EMBED=20 DOWNSAMPLE_FACTOR=2 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=5 KL_MULTIPLIER=1e-6
run_idx=$((run_idx + 1))

launch_run "${run_idx}" "${GPU_LIST[$((run_idx % ${#GPU_LIST[@]}))]}" t24_h96_l24_b4_k5 \
    HIDDEN_DIM=96 N_LATENT_DIMS=24 N_EMBED=20 DOWNSAMPLE_FACTOR=2 BLOCKS_PER_LEVEL=4 KERNEL_SIZE=5 KL_MULTIPLIER=1e-6
run_idx=$((run_idx + 1))

launch_run "${run_idx}" "${GPU_LIST[$((run_idx % ${#GPU_LIST[@]}))]}" t24_h96_l24_b3_k7 \
    HIDDEN_DIM=96 N_LATENT_DIMS=24 N_EMBED=20 DOWNSAMPLE_FACTOR=2 BLOCKS_PER_LEVEL=3 KERNEL_SIZE=7 KL_MULTIPLIER=1e-6

wait
