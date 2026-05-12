#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT_DIR))

from reactive_diffusion_policy.model.vae.temporal_conv_vae import TemporalConvVAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TemporalConvVAE batch-size-1 inference latency."
    )
    parser.add_argument("--ckpt", required=True, help="Path to TemporalConvVAE checkpoint.")
    parser.add_argument("--name", required=True, help="Name used in printed and saved results.")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--n-latent-dims", type=int, required=True)
    parser.add_argument("--n-embed", type=int, default=20)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--blocks-per-level", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--kl-multiplier", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn, warmup, iters, device):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)

        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        wall_start = time.perf_counter()
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            end_event.record()
        synchronize(device)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iters

        cuda_ms = None
        if device.type == "cuda":
            cuda_ms = start_event.elapsed_time(end_event) / iters

    return {"wall_ms": wall_ms, "cuda_ms": cuda_ms}


def main():
    args = parse_args()
    if args.batch_size != 1:
        print(f"Warning: requested batch size is {args.batch_size}; expected batch size is 1.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    shape_meta = {"action": {"shape": [args.action_dim]}}
    model = TemporalConvVAE(
        shape_meta=shape_meta,
        horizon=args.horizon,
        n_latent_dims=args.n_latent_dims,
        hidden_dim=args.hidden_dim,
        downsample_factor=args.downsample_factor,
        blocks_per_level=args.blocks_per_level,
        kernel_size=args.kernel_size,
        kl_multiplier=args.kl_multiplier,
        n_embed=args.n_embed,
        eval=True,
        device=str(device),
        load_dir=str(ckpt_path),
    )
    model.to(device)
    model.eval()

    action = torch.randn(args.batch_size, args.horizon, args.action_dim, device=device)
    quant_latent = torch.randn(
        args.batch_size,
        args.horizon // args.downsample_factor,
        args.n_embed,
        device=device,
    )
    encoded_latent = torch.randn(
        args.batch_size,
        args.horizon // args.downsample_factor,
        args.n_latent_dims,
        device=device,
    )

    results = {
        "name": args.name,
        "ckpt": str(ckpt_path),
        "device": str(device),
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "action_dim": args.action_dim,
        "hidden_dim": args.hidden_dim,
        "n_latent_dims": args.n_latent_dims,
        "n_embed": args.n_embed,
        "downsample_factor": args.downsample_factor,
        "latent_horizon": args.horizon // args.downsample_factor,
        "blocks_per_level": args.blocks_per_level,
        "kernel_size": args.kernel_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "num_params": sum(p.numel() for p in model.optim_params),
        "latency": {},
    }

    results["latency"]["encode_to_latent"] = benchmark(
        lambda: model.encode_to_latent(action),
        args.warmup,
        args.iters,
        device,
    )
    results["latency"]["decode_from_quant_latent"] = benchmark(
        lambda: model.decode_from_latent(quant_latent),
        args.warmup,
        args.iters,
        device,
    )
    results["latency"]["decode_from_encoded_latent"] = benchmark(
        lambda: model.get_action_from_latent(encoded_latent),
        args.warmup,
        args.iters,
        device,
    )
    results["latency"]["encode_then_decode"] = benchmark(
        lambda: model.encode_then_decode({"action": action}),
        args.warmup,
        args.iters,
        device,
    )

    print(json.dumps(results, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
