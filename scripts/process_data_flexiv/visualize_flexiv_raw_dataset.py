"""
python scripts/process_data_flexiv/visualize_flexiv_zarr_dataset.py   --dataset_path /mnt/data/kywang/visual_tactile/flexiv_dataset/Flexiv_pass_car_0429   --num_trajectories 5 --max_video_frames -1
"""


import argparse
import re
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_episodes(dataset_path):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    episodes = sorted(
        hdf5_path.parent
        for hdf5_path in dataset_path.rglob("dataset.hdf5")
        if hdf5_path.is_file()
    )
    if not episodes:
        raise ValueError(f"No episodes with dataset.hdf5 found under {dataset_path}")
    return episodes


def decode_hdf5_path(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def estimate_fps(timestamps, default_fps=30.0):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) <= 1:
        return float(default_fps)
    duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
    if duration_s <= 0:
        return float(default_fps)
    return float((len(timestamps) - 1) / duration_s)


def select_video_indices(num_frames, max_video_frames):
    if max_video_frames is None or max_video_frames < 0 or num_frames <= max_video_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, max_video_frames, dtype=np.int64)


def read_pose_xyz(h5_file, arm):
    pose_path = f"action/{arm}_eef/feedback/pose"
    if pose_path not in h5_file:
        return None, None
    pose_ds = h5_file[pose_path]
    xyz = np.asarray(pose_ds["value"], dtype=np.float32)[:, :3]
    timestamps = np.asarray(pose_ds["timestamp"], dtype=np.float64)
    return xyz, timestamps


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


def plot_trajectory(save_path, episode_name, left_xyz, right_xyz):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if left_xyz is not None and len(left_xyz) > 0:
        ax.plot(left_xyz[:, 0], left_xyz[:, 1], left_xyz[:, 2], label="left_eef", linewidth=2)
        ax.scatter(left_xyz[0, 0], left_xyz[0, 1], left_xyz[0, 2], marker="o", s=40)
        ax.scatter(left_xyz[-1, 0], left_xyz[-1, 1], left_xyz[-1, 2], marker="x", s=50)

    if right_xyz is not None and len(right_xyz) > 0:
        ax.plot(right_xyz[:, 0], right_xyz[:, 1], right_xyz[:, 2], label="right_eef", linewidth=2)
        ax.scatter(right_xyz[0, 0], right_xyz[0, 1], right_xyz[0, 2], marker="o", s=40)
        ax.scatter(right_xyz[-1, 0], right_xyz[-1, 1], right_xyz[-1, 2], marker="x", s=50)

    set_equal_3d_axes(ax, [left_xyz, right_xyz])
    ax.set_title(episode_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def write_camera_video(
    save_path,
    episode_path,
    camera_records,
    frame_indices,
    output_fps,
    resize_width,
    resize_height,
):
    first_record = camera_records[frame_indices[0]]
    first_image_path = episode_path / decode_hdf5_path(first_record["file_path"])
    first_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Failed to read first image: {first_image_path}")

    if resize_width > 0 and resize_height > 0:
        video_size = (resize_width, resize_height)
    else:
        video_size = (first_image.shape[1], first_image.shape[0])

    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        video_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {save_path}")

    for out_idx, frame_idx in enumerate(frame_indices):
        record = camera_records[frame_idx]
        image_path = episode_path / decode_hdf5_path(record["file_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if image.shape[1] != video_size[0] or image.shape[0] != video_size[1]:
            image = cv2.resize(image, video_size, interpolation=cv2.INTER_AREA)

        timestamp = float(record["timestamp"])
        cv2.putText(
            image,
            f"frame {frame_idx} -> {out_idx}, t={timestamp:.1f} ms",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(image)

    writer.release()


def visualize_episode(
    episode_path,
    camera_name,
    save_dir,
    max_video_frames,
    output_fps,
    resize_width,
    resize_height,
):
    hdf5_path = episode_path / "dataset.hdf5"
    with h5py.File(hdf5_path, "r") as h5_file:
        camera_path = f"observation/camera/{camera_name}"
        if camera_path not in h5_file:
            raise KeyError(f"Missing HDF5 key {camera_path} in {hdf5_path}")

        camera_ds = h5_file[camera_path]
        camera_records = camera_ds[:]
        camera_timestamps = np.asarray(camera_records["timestamp"], dtype=np.float64)
        video_indices = select_video_indices(len(camera_records), max_video_frames)
        video_fps = output_fps if output_fps > 0 else estimate_fps(camera_timestamps)

        left_xyz, _ = read_pose_xyz(h5_file, "left")
        right_xyz, _ = read_pose_xyz(h5_file, "right")

    episode_save_dir = save_dir / safe_name(episode_path.name)
    episode_save_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory(
        episode_save_dir / "trajectory_3d.png",
        episode_path.name,
        left_xyz,
        right_xyz,
    )
    write_camera_video(
        episode_save_dir / f"{camera_name}.mp4",
        episode_path,
        camera_records,
        video_indices,
        video_fps,
        resize_width,
        resize_height,
    )

    return {
        "episode": str(episode_path),
        "camera_frames": len(camera_records),
        "video_frames": len(video_indices),
        "video_fps": video_fps,
        "left_pose_frames": 0 if left_xyz is None else len(left_xyz),
        "right_pose_frames": 0 if right_xyz is None else len(right_xyz),
        "output_dir": str(episode_save_dir),
    }


def write_summary(save_dir, args, summaries):
    lines = [
        "Flexiv raw dataset visualization",
        "",
        f"dataset_path: {args.dataset_path}",
        f"camera_name: {args.camera_name}",
        f"num_trajectories: {args.num_trajectories}",
        f"max_video_frames: {args.max_video_frames}",
        "",
        "episodes:",
    ]
    for idx, item in enumerate(summaries):
        lines.extend(
            [
                f"  [{idx}] {item['episode']}",
                f"      camera_frames: {item['camera_frames']}",
                f"      video_frames: {item['video_frames']}",
                f"      video_fps: {item['video_fps']:.6f}",
                f"      left_pose_frames: {item['left_pose_frames']}",
                f"      right_pose_frames: {item['right_pose_frames']}",
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
        help="Raw Flexiv dataset path. The script recursively searches for dataset.hdf5.",
    )
    parser.add_argument("--camera_name", type=str, required=True)
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=5,
        help="Number of episodes/trajectories to visualize.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/process_data/flexiv_raw_vis",
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=600,
        help="Max sampled frames per video. Use -1 to save all frames.",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=30.0,
        help="Output video fps. Use <=0 to estimate from camera timestamps.",
    )
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--resize_height", type=int, default=480)
    args = parser.parse_args()

    episodes = find_episodes(args.dataset_path)
    if args.num_trajectories > 0:
        episodes = episodes[:args.num_trajectories]

    save_dir = Path(args.save_path) / safe_name(Path(args.dataset_path).name) / args.camera_name
    save_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for episode_path in tqdm.tqdm(episodes, desc="Visualizing raw Flexiv episodes"):
        summaries.append(
            visualize_episode(
                episode_path=episode_path,
                camera_name=args.camera_name,
                save_dir=save_dir,
                max_video_frames=args.max_video_frames,
                output_fps=args.output_fps,
                resize_width=args.resize_width,
                resize_height=args.resize_height,
            )
        )

    write_summary(save_dir, args, summaries)
    print(f"Saved visualization to: {save_dir}")
    print(f"Summary: {save_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
