"""
python scripts/process_data_flexiv/visualize_flexiv_zarr_dataset.py   --dataset_path /mnt/data/kywang/visual_tactile/flexiv_dataset/Flexiv_clothing_3_packing_0429   --num_trajectories 2 --max_video_frames -1

python scripts/process_data_flexiv/visualize_flexiv_zarr_dataset.py   --dataset_path /mnt/data/kywang/visual_tactile/flexiv_dataset/Flexiv_pass_car_0429   --num_trajectories 2 --max_video_frames -1
"""

import argparse
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tqdm
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def resolve_zarr_path(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.name == "replay_buffer.zarr":
        zarr_path = dataset_path
    else:
        zarr_path = dataset_path / "replay_buffer.zarr"
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Cannot find replay_buffer.zarr: {zarr_path}")
    return zarr_path


def select_video_indices(num_frames, max_video_frames):
    if max_video_frames is None or max_video_frames < 0 or num_frames <= max_video_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, max_video_frames, dtype=np.int64)


def get_episode_bounds(episode_ends, episode_idx):
    end = int(episode_ends[episode_idx])
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    return start, end


def set_equal_3d_axes(ax, points):
    valid_points = [pts for pts in points if pts is not None and len(pts) > 0]
    if not valid_points:
        return
    all_points = np.concatenate(valid_points, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def get_xyz(zarr_data, key, start, end):
    if key not in zarr_data:
        return None
    arr = np.asarray(zarr_data[key][start:end], dtype=np.float32)
    if arr.shape[1] < 3:
        return None
    return arr[:, :3]


def plot_trajectory(save_path, title, left_xyz, right_xyz):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if left_xyz is not None and len(left_xyz) > 0:
        ax.plot(left_xyz[:, 0], left_xyz[:, 1], left_xyz[:, 2], label="left_robot_tcp_pose", linewidth=2)
        ax.scatter(left_xyz[0, 0], left_xyz[0, 1], left_xyz[0, 2], marker="o", s=40)
        ax.scatter(left_xyz[-1, 0], left_xyz[-1, 1], left_xyz[-1, 2], marker="x", s=50)

    if right_xyz is not None and len(right_xyz) > 0:
        ax.plot(right_xyz[:, 0], right_xyz[:, 1], right_xyz[:, 2], label="right_robot_tcp_pose", linewidth=2)
        ax.scatter(right_xyz[0, 0], right_xyz[0, 1], right_xyz[0, 2], marker="o", s=40)
        ax.scatter(right_xyz[-1, 0], right_xyz[-1, 1], right_xyz[-1, 2], marker="x", s=50)

    set_equal_3d_axes(ax, [left_xyz, right_xyz])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_z_position(save_path, title, left_xyz, right_xyz):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    if left_xyz is not None and len(left_xyz) > 0:
        left_z = left_xyz[:, 2]
        axes[0].plot(left_z, label="left z")
        axes[1].plot(np.abs(np.diff(left_z, prepend=left_z[0])), label="abs diff left z")

    if right_xyz is not None and len(right_xyz) > 0:
        right_z = right_xyz[:, 2]
        axes[0].plot(right_z, label="right z")
        axes[1].plot(np.abs(np.diff(right_z, prepend=right_z[0])), label="abs diff right z")

    axes[0].set_title(title)
    axes[0].set_ylabel("z position")
    axes[1].set_ylabel("abs z diff")
    axes[1].set_xlabel("frame")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def write_image_video(
    save_path,
    images,
    frame_indices,
    global_start,
    output_fps,
    resize_width,
    resize_height,
):
    first = np.asarray(images[0])
    if resize_width > 0 and resize_height > 0:
        video_size = (resize_width, resize_height)
    else:
        video_size = (first.shape[1], first.shape[0])

    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        video_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {save_path}")

    for out_idx, frame_idx in enumerate(frame_indices):
        image_rgb = np.asarray(images[frame_idx], dtype=np.uint8)
        if image_rgb.shape[1] != video_size[0] or image_rgb.shape[0] != video_size[1]:
            image_rgb = cv2.resize(image_rgb, video_size, interpolation=cv2.INTER_AREA)
        image_bgr = image_rgb[..., ::-1].copy()
        cv2.putText(
            image_bgr,
            f"episode_frame={frame_idx}, global_frame={global_start + int(frame_idx)}, video_frame={out_idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(image_bgr)

    writer.release()


def infer_image_keys(zarr_data):
    preferred = ["left_wrist_img", "right_wrist_img"]
    keys = [key for key in preferred if key in zarr_data]
    if keys:
        return keys
    return [
        key for key, value in zarr_data.items()
        if len(value.shape) == 4 and value.dtype == np.dtype("uint8")
    ]


def visualize_episode(
    zarr_data,
    episode_idx,
    start,
    end,
    image_keys,
    save_dir,
    max_video_frames,
    output_fps,
    resize_width,
    resize_height,
):
    episode_len = end - start
    if episode_len <= 0:
        raise ValueError(f"Invalid episode bounds: start={start}, end={end}")

    episode_save_dir = save_dir / f"episode_{episode_idx:04d}_frames_{start}_{end}"
    episode_save_dir.mkdir(parents=True, exist_ok=True)

    left_xyz = get_xyz(zarr_data, "left_robot_tcp_pose", start, end)
    right_xyz = get_xyz(zarr_data, "right_robot_tcp_pose", start, end)
    title = f"episode {episode_idx}, frames [{start}, {end}), length={episode_len}"
    plot_trajectory(episode_save_dir / "trajectory_3d.png", title, left_xyz, right_xyz)
    plot_z_position(episode_save_dir / "z_position.png", title, left_xyz, right_xyz)

    frame_indices = select_video_indices(episode_len, max_video_frames)
    for image_key in image_keys:
        if image_key not in zarr_data:
            raise KeyError(f"Missing image key in zarr data: {image_key}")
        images = zarr_data[image_key][start:end]
        write_image_video(
            episode_save_dir / f"{image_key}.mp4",
            images,
            frame_indices,
            global_start=start,
            output_fps=output_fps,
            resize_width=resize_width,
            resize_height=resize_height,
        )

    return {
        "episode_idx": episode_idx,
        "start": start,
        "end": end,
        "length": episode_len,
        "video_frames": len(frame_indices),
        "image_keys": image_keys,
        "output_dir": str(episode_save_dir),
    }


def write_summary(save_dir, args, zarr_path, zarr_data, episode_ends, summaries):
    lines = [
        "Flexiv processed zarr dataset visualization",
        "",
        f"dataset_path: {args.dataset_path}",
        f"zarr_path: {zarr_path}",
        f"num_episodes_total: {len(episode_ends)}",
        f"num_trajectories_visualized: {len(summaries)}",
        f"image_keys: {', '.join(args.image_keys) if args.image_keys else 'auto'}",
        f"max_video_frames: {args.max_video_frames}",
        f"output_fps: {args.output_fps}",
        "",
        "zarr data keys:",
    ]
    for key in sorted(zarr_data.keys()):
        arr = zarr_data[key]
        lines.append(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    lines.extend(["", "episodes:"])
    for item in summaries:
        lines.extend(
            [
                f"  [{item['episode_idx']}] frames=[{item['start']}, {item['end']}), length={item['length']}",
                f"      video_frames: {item['video_frames']}",
                f"      image_keys: {', '.join(item['image_keys'])}",
                f"      output_dir: {item['output_dir']}",
            ]
        )
    (save_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Processed dataset directory containing replay_buffer.zarr, or replay_buffer.zarr itself.",
    )
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=None,
        help="Image keys to render. Defaults to left_wrist_img/right_wrist_img if present.",
    )
    parser.add_argument("--num_trajectories", type=int, default=5)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/process_data/flexiv_zarr_vis",
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=600,
        help="Max sampled frames per video. Use -1 to save all frames.",
    )
    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--resize_height", type=int, default=480)
    args = parser.parse_args()

    zarr_path = resolve_zarr_path(args.dataset_path)
    zarr_root = zarr.open(str(zarr_path), mode="r")
    zarr_data = zarr_root["data"]
    episode_ends = np.asarray(zarr_root["meta/episode_ends"], dtype=np.int64)

    image_keys = args.image_keys if args.image_keys is not None else infer_image_keys(zarr_data)
    if not image_keys:
        raise ValueError("No image keys found. Please pass --image_keys explicitly.")

    end_episode = len(episode_ends)
    if args.num_trajectories > 0:
        end_episode = min(end_episode, args.start_episode + args.num_trajectories)
    episode_indices = list(range(args.start_episode, end_episode))

    save_dir = Path(args.save_path) / safe_name(Path(args.dataset_path).name)
    save_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for episode_idx in tqdm.tqdm(episode_indices, desc="Visualizing processed zarr episodes"):
        start, end = get_episode_bounds(episode_ends, episode_idx)
        summaries.append(
            visualize_episode(
                zarr_data=zarr_data,
                episode_idx=episode_idx,
                start=start,
                end=end,
                image_keys=image_keys,
                save_dir=save_dir,
                max_video_frames=args.max_video_frames,
                output_fps=args.output_fps,
                resize_width=args.resize_width,
                resize_height=args.resize_height,
            )
        )

    write_summary(save_dir, args, zarr_path, zarr_data, episode_ends, summaries)
    print(f"Saved visualization to: {save_dir}")
    print(f"Summary: {save_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
