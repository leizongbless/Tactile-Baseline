#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

GPU_ID=${GPU_ID:-1}
CONDA_ENV=${CONDA_ENV:-rdp}
batch_size=${batch_size:-1}
horizon=${horizon:-48}
warmup=${warmup:-200}
iters=${iters:-2000}

CKPT_ROOT=${CKPT_ROOT:-"/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/flexiv_clothing_3_packing_0429_30hz/ckpts_abs/conv_vae/n_embed_20"}
RESULT_DIR=${RESULT_DIR:-"${ROOT_DIR}/data/outputs/latency/temporal_conv_vae_cloth0429_h${horizon}"}

if [ -f "/home/kywang/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/kywang/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

cd "${ROOT_DIR}"
mkdir -p "${RESULT_DIR}"

run_case() {
    local name="$1"
    local hidden_dim="$2"
    local n_latent_dims="$3"
    local n_embed="$4"
    local downsample_factor="$5"
    local blocks_per_level="$6"
    local kernel_size="$7"
    local kl_multiplier="$8"
    local ckpt="${CKPT_ROOT}/${name}/latest.ckpt"

    if [ ! -f "${ckpt}" ]; then
        echo "Skip ${name}: checkpoint not found at ${ckpt}"
        return 0
    fi

    echo ""
    echo "Benchmark ${name}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${SCRIPT_DIR}/benchmark_temporal_conv_vae_latency.py" \
        --name "${name}" \
        --ckpt "${ckpt}" \
        --horizon "${horizon}" \
        --action-dim 20 \
        --hidden-dim "${hidden_dim}" \
        --n-latent-dims "${n_latent_dims}" \
        --n-embed "${n_embed}" \
        --downsample-factor "${downsample_factor}" \
        --blocks-per-level "${blocks_per_level}" \
        --kernel-size "${kernel_size}" \
        --kl-multiplier "${kl_multiplier}" \
        --batch-size "${batch_size}" \
        --warmup "${warmup}" \
        --iters "${iters}" \
        --device cuda \
        --output-json "${RESULT_DIR}/${name}.json"
}

run_case "h128_l24_kl1e6_epoch300" 128 24 20 2 3 5 1e-6
run_case "h128_l16_kl3e6_epoch300" 128 16 20 2 3 5 3e-6
run_case "h128_l24_kl3e6_epoch300" 128 24 20 2 3 5 3e-6
run_case "h160_l24_kl3e6_epoch300" 160 24 20 2 3 5 3e-6
run_case "h128_l24_kl1e6_epoch600" 128 24 20 2 3 5 1e-6

echo ""
echo "Latency results saved to ${RESULT_DIR}"
