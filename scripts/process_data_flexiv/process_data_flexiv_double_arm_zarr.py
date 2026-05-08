import argparse
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import tqdm

from process_data_flexiv_zarr import (
    append_images_from_paths,
    append_to_zarr_dataset,
    decode_hdf5_path,
    estimate_fps,
    find_start_end_by_multi_zdiff,
    get_aligned_indices,
    pose_quat_to_tcp_pose_9d,
    select_indices_by_fps,
)


def get_camera_indices(h5_file, camera_name, main_indices, main_timestamps):
    camera_path = f"observation/camera/{camera_name}"
    if camera_path not in h5_file:
        raise KeyError(f"Missing HDF5 key {camera_path}")

    camera_ds = h5_file[camera_path]
    camera_timestamps = np.asarray(camera_ds["timestamp"], dtype=np.float64)
    map_path = f"meta/index_map/observation/camera/{camera_name}"
    return get_aligned_indices(
        h5_file,
        map_path,
        main_indices,
        main_timestamps,
        camera_timestamps,
    )


def read_arm_state(h5_file, arm, main_indices, main_timestamps, position_scale):
    pose_path = f"action/{arm}_eef/feedback/pose"
    gripper_path = f"action/{arm}_eef/feedback/gripper"
    for required_path in [pose_path, gripper_path]:
        if required_path not in h5_file:
            raise KeyError(f"Missing HDF5 key {required_path}")

    pose_ds = h5_file[pose_path]
    gripper_ds = h5_file[gripper_path]
    pose_timestamps = np.asarray(pose_ds["timestamp"], dtype=np.float64)
    gripper_timestamps = np.asarray(gripper_ds["timestamp"], dtype=np.float64)

    pose_indices = get_aligned_indices(
        h5_file,
        f"meta/index_map/action/{arm}_eef/feedback/pose",
        main_indices,
        main_timestamps,
        pose_timestamps,
    )
    gripper_indices = get_aligned_indices(
        h5_file,
        f"meta/index_map/action/{arm}_eef/feedback/gripper",
        main_indices,
        main_timestamps,
        gripper_timestamps,
    )

    tcp_pose = pose_quat_to_tcp_pose_9d(
        pose_ds[pose_indices]["value"], position_scale=position_scale
    )
    gripper = np.asarray(gripper_ds[gripper_indices]["value"], dtype=np.float32)[:, None]
    state_with_gripper = np.concatenate([tcp_pose, gripper], axis=1)

    if len(state_with_gripper) > 1:
        action = np.concatenate(
            [state_with_gripper[1:], state_with_gripper[-1:]],
            axis=0,
        )
    else:
        action = state_with_gripper.copy()

    return (
        tcp_pose.astype(np.float32, copy=False),
        gripper.astype(np.float32, copy=False),
        action.astype(np.float32, copy=False),
    )


def read_camera_paths(h5_file, episode_path, camera_name, camera_indices):
    camera_ds = h5_file[f"observation/camera/{camera_name}"]
    camera_records = camera_ds[camera_indices]
    return [
        episode_path / decode_hdf5_path(file_path)
        for file_path in camera_records["file_path"]
    ]


def build_next_step_action(tcp_pose, gripper):
    state_with_gripper = np.concatenate([tcp_pose, gripper], axis=1)
    if len(state_with_gripper) > 1:
        return np.concatenate(
            [state_with_gripper[1:], state_with_gripper[-1:]],
            axis=0,
        ).astype(np.float32, copy=False)
    return state_with_gripper.astype(np.float32, copy=False)


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
            hdf5_path.parent
            for hdf5_path in data_dir.rglob("dataset.hdf5")
            if hdf5_path.is_file()
        )
        if not episodes:
            raise ValueError(f"No Flexiv episodes found in {data_dir}")

        episode_list.extend(episodes)
        resolved_tasks.append(data_dir.name)

    if episode_length != -1:
        episode_list = episode_list[:episode_length]

    output_name = resolved_tasks[0] if len(resolved_tasks) == 1 else "_".join(resolved_tasks)
    return episode_list, output_name


def process_one_episode_double_arm(
    episode_path,
    left_camera_name,
    right_camera_name,
    target_fps,
    position_scale,
    start_z_diff_thresh,
    end_extra_frames,
    max_frames_per_episode=-1,
):
    hdf5_path = episode_path / "dataset.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Missing dataset.hdf5: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5_file:
        left_camera_path = f"observation/camera/{left_camera_name}"
        if left_camera_path not in h5_file:
            raise KeyError(f"Missing HDF5 key {left_camera_path} in {hdf5_path}")

        left_camera_ds = h5_file[left_camera_path]
        left_camera_timestamps = np.asarray(
            left_camera_ds["timestamp"], dtype=np.float64
        )
        main_indices = select_indices_by_fps(left_camera_timestamps, target_fps)
        if max_frames_per_episode > 0:
            main_indices = main_indices[:max_frames_per_episode]
        if len(main_indices) == 0:
            raise ValueError(f"No frames selected for {episode_path}")

        right_camera_indices = get_camera_indices(
            h5_file,
            right_camera_name,
            main_indices,
            left_camera_timestamps,
        )

        left_tcp_pose, left_gripper, _ = read_arm_state(
            h5_file,
            "left",
            main_indices,
            left_camera_timestamps,
            position_scale=position_scale,
        )
        right_tcp_pose, right_gripper, _ = read_arm_state(
            h5_file,
            "right",
            main_indices,
            left_camera_timestamps,
            position_scale=position_scale,
        )

        start_frame, end_frame = find_start_end_by_multi_zdiff(
            [left_tcp_pose, right_tcp_pose],
            z_diff_thresh=start_z_diff_thresh,
            end_extra_frames=end_extra_frames,
        )
        print(
            f"[target_fps={target_fps}] start: {start_frame}, end: {end_frame}, "
            f"raw_frames={len(left_tcp_pose)}, start_z_diff_thresh={start_z_diff_thresh}, "
            f"end_extra_frames={end_extra_frames}"
        )

        left_tcp_pose = left_tcp_pose[start_frame:end_frame]
        left_gripper = left_gripper[start_frame:end_frame]
        right_tcp_pose = right_tcp_pose[start_frame:end_frame]
        right_gripper = right_gripper[start_frame:end_frame]
        main_indices = main_indices[start_frame:end_frame]
        right_camera_indices = right_camera_indices[start_frame:end_frame]

        left_action = build_next_step_action(left_tcp_pose, left_gripper)
        right_action = build_next_step_action(right_tcp_pose, right_gripper)
        action = np.concatenate([left_action, right_action], axis=1)
        left_image_paths = read_camera_paths(
            h5_file, episode_path, left_camera_name, main_indices
        )
        right_image_paths = read_camera_paths(
            h5_file, episode_path, right_camera_name, right_camera_indices
        )

        return {
            "left_robot_tcp_pose": left_tcp_pose,
            "left_robot_gripper_width": left_gripper,
            "right_robot_tcp_pose": right_tcp_pose,
            "right_robot_gripper_width": right_gripper,
            "action": action.astype(np.float32, copy=False),
            "left_image_paths": left_image_paths,
            "right_image_paths": right_image_paths,
            "camera_timestamps": left_camera_timestamps[main_indices],
            "source_fps": estimate_fps(left_camera_timestamps),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "raw_frame_count": len(left_camera_timestamps),
        }


def write_readme(save_data_path, args, total_frames, episode_ends, source_fps_values):
    source_fps_text = "unknown"
    if source_fps_values:
        source_fps_text = (
            f"mean={np.mean(source_fps_values):.6f} Hz, "
            f"min={np.min(source_fps_values):.6f} Hz, "
            f"max={np.max(source_fps_values):.6f} Hz"
        )

    lines = [
        "Flexiv double-arm zarr dataset",
        "",
        f"source_root: {args.root_path}",
        f"task_list: {', '.join(args.task_list)}",
        f"left_camera_name: {args.left_camera_name}",
        f"right_camera_name: {args.right_camera_name}",
        f"target_fps: {args.target_fps} Hz",
        f"source_left_camera_fps: {source_fps_text}",
        f"start_z_diff_thresh: {args.start_z_diff_thresh}",
        f"end_extra_frames: {args.end_extra_frames}",
        f"episode_count: {len(episode_ends)}",
        f"total_frames: {total_frames}",
        f"image_size: {args.image_width}x{args.image_height} (stored as HWC RGB uint8)",
        "",
        "zarr_path: replay_buffer.zarr",
        "",
        "keys:",
        "  data/left_wrist_img: uint8, shape=(N, H, W, 3), RGB image from left camera",
        "  data/right_wrist_img: uint8, shape=(N, H, W, 3), RGB image from right camera",
        "  data/left_robot_tcp_pose: float32, shape=(N, 9), left xyz position + 6D rotation",
        "  data/left_robot_gripper_width: float32, shape=(N, 1), left Flexiv gripper feedback value",
        "  data/right_robot_tcp_pose: float32, shape=(N, 9), right xyz position + 6D rotation",
        "  data/right_robot_gripper_width: float32, shape=(N, 1), right Flexiv gripper feedback value",
        "  data/action: float32, shape=(N, 20), next-step left 10D action followed by right 10D action",
        "  meta/episode_ends: int64, shape=(num_episodes,), cumulative frame ends",
        "",
        "notes:",
        "  Each 10D arm action is xyz + 6D rotation + gripper.",
        "  The first 10 action dims are left arm; the last 10 dims are right arm.",
        "  start/end frames are trimmed by max(abs(diff(left_z)), abs(diff(right_z))).",
        "  tactile keys are intentionally not stored.",
    ]
    (save_data_path / "readme.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def normalize_camera_args(args):
    if args.camera_name is not None:
        if len(args.camera_name) != 2:
            raise ValueError("--camera_name expects exactly 2 values in double-arm mode")
        args.left_camera_name = args.camera_name[0]
        args.right_camera_name = args.camera_name[1]


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
    parser.add_argument("--left_camera_name", type=str, default="ldl_hand_fisheye")
    parser.add_argument("--right_camera_name", type=str, default="rdl_hand_fisheye")
    parser.add_argument(
        "--camera_name",
        nargs="+",
        type=str,
        default=None,
        help="Optional shorthand: pass left and right camera names.",
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        default=["left", "right"],
        help="Accepted for compatibility; double-arm output always uses left then right.",
    )
    parser.add_argument("--episode_length", type=int, default=-1)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--position_scale", type=float, default=1.0)
    parser.add_argument(
        "--start_z_diff_thresh",
        type=float,
        default=0.001,
        help="Trim start/end by max abs(diff(left_z), diff(right_z)); <=0 disables trimming.",
    )
    parser.add_argument(
        "--end_extra_frames",
        type=int,
        default=3,
        help="Extra frames kept after detected motion end.",
    )
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=-1,
        help="Debug option. -1 means keep all selected frames.",
    )
    parser.add_argument("--image_batch_size", type=int, default=128)
    parser.add_argument("--image_num_workers", type=int, default=8)
    parser.add_argument("--stage_in_tmpfs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tmpfs_root", type=str, default="/dev/shm/flexiv_zarr")
    parser.add_argument(
        "--cleanup_tmpfs_after_copy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--profile_timing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    normalize_camera_args(args)

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
    final_save_data_path = Path(args.save_path) / output_name
    if args.stage_in_tmpfs:
        save_data_path = Path(args.tmpfs_root) / f"{output_name}_{os.getpid()}"
        if save_data_path.exists():
            shutil.rmtree(save_data_path)
    else:
        save_data_path = final_save_data_path
    save_zarr_path = save_data_path / "replay_buffer.zarr"
    save_data_path.mkdir(parents=True, exist_ok=True)

    print("Flexiv double-arm processing settings:")
    print(f"  root_path: {args.root_path}")
    print(f"  task_list: {args.task_list}")
    print(f"  final_save_data_path: {final_save_data_path}")
    print(f"  staging_save_data_path: {save_data_path}")
    print(f"  save_zarr_path: {save_zarr_path}")
    print(f"  target_fps: {args.target_fps}")
    print(f"  left_camera_name: {args.left_camera_name}")
    print(f"  right_camera_name: {args.right_camera_name}")
    print(f"  image_num_workers: {args.image_num_workers}")
    print(f"  stage_in_tmpfs: {args.stage_in_tmpfs}")
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
    right_robot_tcp_pose_ds = zarr_data.create_dataset(
        "right_robot_tcp_pose",
        shape=(0, 9),
        chunks=(10000, 9),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    right_robot_gripper_width_ds = zarr_data.create_dataset(
        "right_robot_gripper_width",
        shape=(0, 1),
        chunks=(10000, 1),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    action_ds = zarr_data.create_dataset(
        "action",
        shape=(0, 20),
        chunks=(10000, 20),
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
    right_wrist_img_ds = zarr_data.create_dataset(
        "right_wrist_img",
        shape=(0, args.image_height, args.image_width, 3),
        chunks=(100, args.image_height, args.image_width, 3),
        dtype="uint8",
        overwrite=True,
    )

    episode_end_list = []
    source_fps_values = []
    total_frames = 0
    timing = {
        "process_episode_s": 0.0,
        "append_lowdim_s": 0.0,
        "append_left_image_s": 0.0,
        "append_right_image_s": 0.0,
        "copy_to_final_s": 0.0,
        "total_s": time.perf_counter(),
    }

    for episode_path in tqdm.tqdm(episode_list):
        print(f"loading episode: {episode_path}")
        episode_start_t = time.perf_counter()
        t0 = time.perf_counter()
        episode_data = process_one_episode_double_arm(
            episode_path=episode_path,
            left_camera_name=args.left_camera_name,
            right_camera_name=args.right_camera_name,
            target_fps=args.target_fps,
            position_scale=args.position_scale,
            start_z_diff_thresh=args.start_z_diff_thresh,
            end_extra_frames=args.end_extra_frames,
            max_frames_per_episode=args.max_frames_per_episode,
        )
        timing["process_episode_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        append_to_zarr_dataset(
            left_robot_tcp_pose_ds, episode_data["left_robot_tcp_pose"]
        )
        append_to_zarr_dataset(
            left_robot_gripper_width_ds, episode_data["left_robot_gripper_width"]
        )
        append_to_zarr_dataset(
            right_robot_tcp_pose_ds, episode_data["right_robot_tcp_pose"]
        )
        append_to_zarr_dataset(
            right_robot_gripper_width_ds, episode_data["right_robot_gripper_width"]
        )
        append_to_zarr_dataset(action_ds, episode_data["action"])
        timing["append_lowdim_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        append_images_from_paths(
            left_wrist_img_ds,
            episode_data["left_image_paths"],
            image_size=(args.image_width, args.image_height),
            batch_size=args.image_batch_size,
            num_workers=args.image_num_workers,
        )
        timing["append_left_image_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        append_images_from_paths(
            right_wrist_img_ds,
            episode_data["right_image_paths"],
            image_size=(args.image_width, args.image_height),
            batch_size=args.image_batch_size,
            num_workers=args.image_num_workers,
        )
        timing["append_right_image_s"] += time.perf_counter() - t0

        total_frames += len(episode_data["action"])
        episode_end_list.append(total_frames)
        source_fps_values.append(episode_data["source_fps"])
        if args.profile_timing:
            episode_elapsed = time.perf_counter() - episode_start_t
            print(
                f"episode frames={len(episode_data['action'])}, "
                f"elapsed={episode_elapsed:.3f}s, "
                f"avg={episode_elapsed / max(len(episode_data['action']), 1) * 1000:.3f}ms/frame"
            )

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
    zarr_root.attrs["left_camera_name"] = args.left_camera_name
    zarr_root.attrs["right_camera_name"] = args.right_camera_name
    zarr_root.attrs["arm_order"] = "left,right"
    zarr_root.attrs["start_z_diff_thresh"] = float(args.start_z_diff_thresh)
    zarr_root.attrs["end_extra_frames"] = int(args.end_extra_frames)
    zarr_root.attrs["total_frames"] = int(total_frames)

    write_readme(save_data_path, args, total_frames, episode_ends, source_fps_values)

    if args.stage_in_tmpfs:
        final_save_data_path.parent.mkdir(parents=True, exist_ok=True)
        if final_save_data_path.exists():
            shutil.rmtree(final_save_data_path)
        t0 = time.perf_counter()
        shutil.copytree(save_data_path, final_save_data_path)
        timing["copy_to_final_s"] += time.perf_counter() - t0
        if args.cleanup_tmpfs_after_copy:
            shutil.rmtree(save_data_path)

    timing["total_s"] = time.perf_counter() - timing["total_s"]
    print(f"Finished. Wrote {total_frames} frames to {final_save_data_path / 'replay_buffer.zarr'}")
    print(f"Final dataset: {final_save_data_path}")
    print(f"Readme: {final_save_data_path / 'readme.txt'}")
    if args.profile_timing:
        print("\n=== Profiling Summary ===")
        for key in [
            "process_episode_s",
            "append_lowdim_s",
            "append_left_image_s",
            "append_right_image_s",
            "copy_to_final_s",
            "total_s",
        ]:
            print(f"{key:>20}: {timing[key]:.3f}s")


if __name__ == "__main__":
    main()
