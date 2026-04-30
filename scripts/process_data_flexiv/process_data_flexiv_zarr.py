import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R


def append_to_zarr_dataset(dataset, batch):
    batch = np.asarray(batch)
    prev_len = dataset.shape[0]
    if batch.shape[1:] != dataset.shape[1:]:
        raise ValueError(
            f"Shape mismatch: batch.shape={batch.shape}, dataset.shape={dataset.shape}"
        )
    dataset.resize((prev_len + batch.shape[0],) + dataset.shape[1:])
    dataset[prev_len:prev_len + batch.shape[0]] = batch


def estimate_fps(timestamps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) <= 1:
        return 0.0
    duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
    if duration_s <= 0:
        return 0.0
    return float((len(timestamps) - 1) / duration_s)


def select_indices_by_fps(timestamps, target_fps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) == 0:
        return np.array([], dtype=np.int64)

    source_fps = estimate_fps(timestamps)
    if source_fps > 0 and abs(source_fps - target_fps) / source_fps < 0.02:
        return np.arange(len(timestamps), dtype=np.int64)

    start_time = timestamps[0]
    end_time = timestamps[-1]
    step = 1000.0 / float(target_fps)
    num_out = int(np.floor((end_time - start_time) / step)) + 1
    target_times = start_time + np.arange(num_out, dtype=np.float64) * step

    indices = nearest_indices(target_times, timestamps)
    indices = np.unique(indices)
    return indices.astype(np.int64)


def nearest_indices(query_timestamps, source_timestamps):
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    source_timestamps = np.asarray(source_timestamps, dtype=np.float64)
    right = np.searchsorted(source_timestamps, query_timestamps, side="left")
    right = np.clip(right, 0, len(source_timestamps) - 1)
    left = np.clip(right - 1, 0, len(source_timestamps) - 1)
    choose_right = (
        np.abs(source_timestamps[right] - query_timestamps)
        < np.abs(source_timestamps[left] - query_timestamps)
    )
    return np.where(choose_right, right, left).astype(np.int64)


def get_aligned_indices(h5_file, map_path, camera_indices, camera_timestamps, source_timestamps):
    if map_path in h5_file:
        index_map = h5_file[map_path]
        return np.asarray(index_map[camera_indices]["aligned_index"], dtype=np.int64)
    return nearest_indices(camera_timestamps[camera_indices], source_timestamps)


def pose_quat_to_tcp_pose_9d(pose, position_scale=1.0):
    pose = np.asarray(pose, dtype=np.float32)
    position = pose[:, :3] * float(position_scale)
    quat = pose[:, 3:7]
    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.clip(quat_norm, 1e-8, None)
    rot_mats = R.from_quat(quat).as_matrix()
    rot6d = np.concatenate([rot_mats[:, :, 0], rot_mats[:, :, 1]], axis=1)
    return np.concatenate([position, rot6d.astype(np.float32)], axis=1)


def decode_hdf5_path(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def process_one_episode(
    episode_path,
    camera_name,
    arm,
    target_fps,
    image_size,
    position_scale,
    max_frames_per_episode=-1,
):
    hdf5_path = episode_path / "dataset.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Missing dataset.hdf5: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5_file:
        camera_path = f"observation/camera/{camera_name}"
        pose_path = f"action/{arm}_eef/feedback/pose"
        gripper_path = f"action/{arm}_eef/feedback/gripper"

        for required_path in [camera_path, pose_path, gripper_path]:
            if required_path not in h5_file:
                raise KeyError(f"Missing HDF5 key {required_path} in {hdf5_path}")

        camera_ds = h5_file[camera_path]
        pose_ds = h5_file[pose_path]
        gripper_ds = h5_file[gripper_path]

        camera_timestamps = np.asarray(camera_ds["timestamp"], dtype=np.float64)
        pose_timestamps = np.asarray(pose_ds["timestamp"], dtype=np.float64)
        gripper_timestamps = np.asarray(gripper_ds["timestamp"], dtype=np.float64)

        camera_indices = select_indices_by_fps(camera_timestamps, target_fps)
        if max_frames_per_episode > 0:
            camera_indices = camera_indices[:max_frames_per_episode]
        if len(camera_indices) == 0:
            raise ValueError(f"No frames selected for {episode_path}")

        pose_indices = get_aligned_indices(
            h5_file,
            f"meta/index_map/action/{arm}_eef/feedback/pose",
            camera_indices,
            camera_timestamps,
            pose_timestamps,
        )
        gripper_indices = get_aligned_indices(
            h5_file,
            f"meta/index_map/action/{arm}_eef/feedback/gripper",
            camera_indices,
            camera_timestamps,
            gripper_timestamps,
        )

        camera_records = camera_ds[camera_indices]
        pose_records = pose_ds[pose_indices]
        gripper_records = gripper_ds[gripper_indices]

        image_paths = [
            episode_path / decode_hdf5_path(file_path)
            for file_path in camera_records["file_path"]
        ]
        tcp_pose = pose_quat_to_tcp_pose_9d(
            pose_records["value"], position_scale=position_scale
        )
        gripper = np.asarray(gripper_records["value"], dtype=np.float32)[:, None]

        state_with_gripper = np.concatenate([tcp_pose, gripper], axis=1)
        if len(state_with_gripper) > 1:
            action = np.concatenate(
                [state_with_gripper[1:], state_with_gripper[-1:]],
                axis=0,
            )
        else:
            action = state_with_gripper.copy()

        return {
            "left_robot_tcp_pose": tcp_pose.astype(np.float32, copy=False),
            "left_robot_gripper_width": gripper.astype(np.float32, copy=False),
            "action": action.astype(np.float32, copy=False),
            "image_paths": image_paths,
            "camera_timestamps": camera_timestamps[camera_indices],
            "source_fps": estimate_fps(camera_timestamps),
            "image_size": image_size,
        }


def append_images_from_paths(dataset, image_paths, image_size, batch_size=128):
    width, height = image_size
    prev_len = dataset.shape[0]
    dataset.resize((prev_len + len(image_paths),) + dataset.shape[1:])

    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch = np.empty((end - start, height, width, 3), dtype=np.uint8)
        for local_idx, image_path in enumerate(image_paths[start:end]):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            if image_bgr.shape[1] != width or image_bgr.shape[0] != height:
                image_bgr = cv2.resize(
                    image_bgr, (width, height), interpolation=cv2.INTER_AREA
                )
            batch[local_idx] = image_bgr[..., ::-1]
        dataset[prev_len + start:prev_len + end] = batch


def resolve_episode_list(root_path, task_list, episode_length):
    root_path = Path(root_path)
    episode_list = []
    resolved_tasks = []

    for task in task_list:
        data_dir = root_path / task
        if not data_dir.exists() and (root_path / "dataset.hdf5").exists():
            data_dir = root_path
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {data_dir}")

        episodes = sorted(
            path for path in data_dir.iterdir()
            if path.is_dir() and (path / "dataset.hdf5").is_file()
        )
        if not episodes and (data_dir / "dataset.hdf5").is_file():
            episodes = [data_dir]
        if not episodes:
            raise ValueError(f"No Flexiv episodes found in {data_dir}")

        episode_list.extend(episodes)
        resolved_tasks.append(data_dir.name)

    if episode_length != -1:
        episode_list = episode_list[:episode_length]

    output_name = resolved_tasks[0] if len(resolved_tasks) == 1 else "_".join(resolved_tasks)
    return episode_list, output_name


def write_readme(save_data_path, args, total_frames, episode_ends, source_fps_values):
    source_fps_text = "unknown"
    if source_fps_values:
        source_fps_text = (
            f"mean={np.mean(source_fps_values):.6f} Hz, "
            f"min={np.min(source_fps_values):.6f} Hz, "
            f"max={np.max(source_fps_values):.6f} Hz"
        )

    lines = [
        "Flexiv zarr dataset",
        "",
        f"source_root: {args.root_path}",
        f"task_list: {', '.join(args.task_list)}",
        f"camera_name: {args.camera_name}",
        f"arm_source: {args.arm}",
        f"target_fps: {args.target_fps} Hz",
        f"source_camera_fps: {source_fps_text}",
        f"episode_count: {len(episode_ends)}",
        f"total_frames: {total_frames}",
        f"image_size: {args.image_width}x{args.image_height} (stored as HWC RGB uint8)",
        "",
        "zarr_path: replay_buffer.zarr",
        "",
        "keys:",
        "  data/left_wrist_img: uint8, shape=(N, H, W, 3), RGB image from ldl_hand_fisheye by default",
        "  data/left_robot_tcp_pose: float32, shape=(N, 9), xyz position + 6D rotation",
        "  data/left_robot_gripper_width: float32, shape=(N, 1), Flexiv gripper feedback value",
        "  data/action: float32, shape=(N, 10), next-step left_robot_tcp_pose + gripper",
        "  meta/episode_ends: int64, shape=(num_episodes,), cumulative frame ends",
        "",
        "notes:",
        "  tactile keys are intentionally not stored.",
        "  key names match the xarm dataset convention for image-only training configs.",
    ]
    readme_path = save_data_path / "readme.txt"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        type=str,
        default="/mnt/data/data/ymm/dataset/Flexiv/release",
        help="Directory containing Flexiv task folders.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/data/kywang/visual_tactile/flexiv_dataset",
        help="Directory where processed task folders are written.",
    )
    parser.add_argument(
        "--task_list",
        nargs="+",
        type=str,
        default=["Flexiv_pass_car_0429"],
        help="Flexiv task folder names under root_path.",
    )
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--camera_name", type=str, default="ldl_hand_fisheye")
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--episode_length", type=int, default=-1)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--position_scale", type=float, default=1.0)
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=-1,
        help="Debug option. -1 means keep all selected frames.",
    )
    parser.add_argument("--image_batch_size", type=int, default=128)
    args = parser.parse_args()

    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "The Flexiv converter requires zarr. Please run it in the same "
            "Python environment used by the original xarm processing script, "
            "or install zarr in the current environment."
        ) from exc

    episode_list, output_name = resolve_episode_list(
        args.root_path, args.task_list, args.episode_length
    )
    save_data_path = Path(args.save_path) / output_name
    save_zarr_path = save_data_path / "replay_buffer.zarr"
    save_data_path.mkdir(parents=True, exist_ok=True)

    print("Flexiv processing settings:")
    print(f"  root_path: {args.root_path}")
    print(f"  task_list: {args.task_list}")
    print(f"  save_data_path: {save_data_path}")
    print(f"  save_zarr_path: {save_zarr_path}")
    print(f"  target_fps: {args.target_fps}")
    print(f"  camera_name: {args.camera_name}")
    print(f"  arm: {args.arm}")
    print(f"  episode_count: {len(episode_list)}")

    zarr_root = zarr.open_group(str(save_zarr_path), mode="w")
    zarr_data = zarr_root.create_group("data", overwrite=True)
    zarr_meta = zarr_root.create_group("meta", overwrite=True)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    left_robot_tcp_pose_ds = zarr_data.create_dataset(
        "left_robot_tcp_pose",
        shape=(0, 9),
        chunks=(10000, 9),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    left_robot_gripper_width_ds = zarr_data.create_dataset(
        "left_robot_gripper_width",
        shape=(0, 1),
        chunks=(10000, 1),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    action_ds = zarr_data.create_dataset(
        "action",
        shape=(0, 10),
        chunks=(10000, 10),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    left_wrist_img_ds = zarr_data.create_dataset(
        "left_wrist_img",
        shape=(0, args.image_height, args.image_width, 3),
        chunks=(100, args.image_height, args.image_width, 3),
        dtype="uint8",
        overwrite=True,
    )

    episode_end_list = []
    source_fps_values = []
    total_frames = 0

    for episode_path in tqdm.tqdm(episode_list):
        print(f"loading episode: {episode_path}")
        episode_data = process_one_episode(
            episode_path=episode_path,
            camera_name=args.camera_name,
            arm=args.arm,
            target_fps=args.target_fps,
            image_size=(args.image_width, args.image_height),
            position_scale=args.position_scale,
            max_frames_per_episode=args.max_frames_per_episode,
        )

        append_to_zarr_dataset(
            left_robot_tcp_pose_ds, episode_data["left_robot_tcp_pose"]
        )
        append_to_zarr_dataset(
            left_robot_gripper_width_ds, episode_data["left_robot_gripper_width"]
        )
        append_to_zarr_dataset(action_ds, episode_data["action"])
        append_images_from_paths(
            left_wrist_img_ds,
            episode_data["image_paths"],
            image_size=(args.image_width, args.image_height),
            batch_size=args.image_batch_size,
        )

        total_frames += len(episode_data["action"])
        episode_end_list.append(total_frames)
        source_fps_values.append(episode_data["source_fps"])

    if total_frames == 0:
        raise ValueError("No valid frames were written.")

    episode_ends = np.asarray(episode_end_list, dtype=np.int64)
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        chunks=(min(10000, len(episode_ends)),),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    zarr_root.attrs["target_fps"] = float(args.target_fps)
    zarr_root.attrs["camera_name"] = args.camera_name
    zarr_root.attrs["arm_source"] = args.arm
    zarr_root.attrs["total_frames"] = int(total_frames)

    write_readme(save_data_path, args, total_frames, episode_ends, source_fps_values)
    print(f"Finished. Wrote {total_frames} frames to {save_zarr_path}")
    print(f"Readme: {save_data_path / 'readme.txt'}")


if __name__ == "__main__":
    main()
