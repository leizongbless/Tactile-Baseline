import os
import pickle
import argparse
import numpy as np
from pprint import pprint


REQUIRED_FILES = [
    "image.pkl",
    "gripper.pkl",
    "state.pkl",
    "tactile.pkl",
]

OPTIONAL_FILES = [
    "force.pkl",
    "tac_force.pkl",
    "tactile_raw.pkl",
]


def safe_load_pickle(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def to_numpy(x):
    try:
        return np.array(x)
    except Exception:
        return None


def check_nonempty_1d_array(name, arr, problems):
    arr = to_numpy(arr)
    if arr is None:
        problems.append(f"{name}: cannot convert to numpy array")
        return None
    if arr.ndim == 0:
        problems.append(f"{name}: is scalar, expected 1D array")
        return arr
    if len(arr) == 0:
        problems.append(f"{name}: empty array")
    return arr


def check_monotonic(name, arr, problems):
    if arr is None or len(arr) <= 1:
        return
    diff = np.diff(arr)
    if np.any(diff < 0):
        problems.append(f"{name}: timestamps not non-decreasing")


def check_same_length(name_a, a, name_b, b, problems):
    if a is None or b is None:
        return
    try:
        la = len(a)
        lb = len(b)
        if la != lb:
            problems.append(f"{name_a} length ({la}) != {name_b} length ({lb})")
    except Exception:
        problems.append(f"cannot compare length of {name_a} and {name_b}")


def sync_and_truncate_timestamps_check(ts_dict):
    """
    只用于检查，不做实际处理保存。
    """
    starts = [arr[0] for arr in ts_dict.values()]
    max_start = max(starts)

    camera1 = ts_dict['camera1']
    begin_idx = np.searchsorted(camera1, max_start, side='left')
    if begin_idx >= len(camera1):
        raise ValueError("max_start > all camera1 timestamps")
    begin = camera1[begin_idx]

    start_indices = {}
    for k, arr in ts_dict.items():
        if k == 'camera1':
            idx = np.searchsorted(arr, begin, side='left')
        else:
            idx = np.abs(arr - begin).argmin()
        start_indices[k] = idx

    ends = [arr[-1] for arr in ts_dict.values()]
    min_end = min(ends)

    end_idx = np.searchsorted(camera1, min_end, side='right') - 1
    if end_idx < 0:
        raise ValueError("min_end < all camera1 timestamps")
    end = camera1[end_idx]

    cut_time = end + 60
    end_indices = {}
    for k, arr in ts_dict.items():
        idx = np.searchsorted(arr, cut_time, side='right')
        end_indices[k] = idx

    processed_lens = {}
    for k, arr in ts_dict.items():
        s, e = start_indices[k], end_indices[k]
        processed_lens[k] = max(0, e - s)

    return {
        "begin": begin,
        "end": end,
        "start_indices": start_indices,
        "end_indices": end_indices,
        "processed_lens": processed_lens,
    }


def check_image_pkl(obj, problems, stats):
    if not isinstance(obj, dict):
        problems.append("image.pkl: root is not dict")
        return

    for cam in ["camera1", "camera2"]:
        if cam not in obj:
            problems.append(f"image.pkl: missing key '{cam}'")
            continue

        cam_obj = obj[cam]
        if not isinstance(cam_obj, dict):
            problems.append(f"image.pkl[{cam}]: not dict")
            continue

        if "image" not in cam_obj:
            problems.append(f"image.pkl[{cam}]: missing 'image'")
        if "timestamps" not in cam_obj:
            problems.append(f"image.pkl[{cam}]: missing 'timestamps'")
            continue

        images = cam_obj.get("image", None)
        ts = check_nonempty_1d_array(f"image.pkl[{cam}]['timestamps']", cam_obj["timestamps"], problems)
        check_monotonic(f"image.pkl[{cam}]['timestamps']", ts, problems)

        if images is None:
            continue

        try:
            img_len = len(images)
            stats[f"{cam}_image_len"] = img_len
            if ts is not None:
                check_same_length(f"{cam}.image", images, f"{cam}.timestamps", ts, problems)
        except Exception:
            problems.append(f"image.pkl[{cam}]['image']: has no valid length")

        if img_len > 0:
            try:
                first_img = np.array(images[0])
                stats[f"{cam}_image_shape0"] = tuple(first_img.shape)
                stats[f"{cam}_image_dtype0"] = str(first_img.dtype)
            except Exception:
                problems.append(f"image.pkl[{cam}]['image'][0]: cannot inspect first image")


def check_tactile_pkl(obj, problems, stats):
    if not isinstance(obj, dict):
        problems.append("tactile.pkl: root is not dict")
        return

    for tac in ["tactile1", "tactile2"]:
        if tac not in obj:
            problems.append(f"tactile.pkl: missing key '{tac}'")
            continue

        tac_obj = obj[tac]
        if not isinstance(tac_obj, dict):
            problems.append(f"tactile.pkl[{tac}]: not dict")
            continue

        if "timestamps" not in tac_obj:
            problems.append(f"tactile.pkl[{tac}]: missing 'timestamps'")
            continue
        if "deform" not in tac_obj:
            problems.append(f"tactile.pkl[{tac}]: missing 'deform'")
            continue

        ts = check_nonempty_1d_array(f"tactile.pkl[{tac}]['timestamps']", tac_obj["timestamps"], problems)
        deform = to_numpy(tac_obj["deform"])

        check_monotonic(f"tactile.pkl[{tac}]['timestamps']", ts, problems)

        if deform is None:
            problems.append(f"tactile.pkl[{tac}]['deform']: cannot convert to numpy")
            continue

        stats[f"{tac}_deform_shape"] = tuple(deform.shape)
        stats[f"{tac}_deform_dtype"] = str(deform.dtype)

        if ts is not None:
            check_same_length(f"{tac}.deform", deform, f"{tac}.timestamps", ts, problems)


def check_state_pkl(obj, problems, stats):
    if not isinstance(obj, dict):
        problems.append("state.pkl: root is not dict")
        return

    if "eef_pose" not in obj:
        problems.append("state.pkl: missing 'eef_pose'")
    if "timestamps" not in obj:
        problems.append("state.pkl: missing 'timestamps'")
        return

    ts = check_nonempty_1d_array("state.pkl['timestamps']", obj["timestamps"], problems)
    check_monotonic("state.pkl['timestamps']", ts, problems)

    eef_pose = to_numpy(obj.get("eef_pose", None))
    if eef_pose is None:
        problems.append("state.pkl['eef_pose']: cannot convert to numpy")
        return

    stats["eef_pose_shape"] = tuple(eef_pose.shape)
    stats["eef_pose_dtype"] = str(eef_pose.dtype)

    if ts is not None:
        check_same_length("eef_pose", eef_pose, "state.timestamps", ts, problems)


def check_gripper_pkl(obj, problems, stats):
    if not isinstance(obj, dict):
        problems.append("gripper.pkl: root is not dict")
        return

    if "gripper_pos" not in obj:
        problems.append("gripper.pkl: missing 'gripper_pos'")
    if "timestamps" not in obj:
        problems.append("gripper.pkl: missing 'timestamps'")
        return

    ts = check_nonempty_1d_array("gripper.pkl['timestamps']", obj["timestamps"], problems)
    check_monotonic("gripper.pkl['timestamps']", ts, problems)

    gp = to_numpy(obj.get("gripper_pos", None))
    if gp is None:
        problems.append("gripper.pkl['gripper_pos']: cannot convert to numpy")
        return

    stats["gripper_pos_shape"] = tuple(gp.shape)
    stats["gripper_pos_dtype"] = str(gp.dtype)

    if ts is not None:
        check_same_length("gripper_pos", gp, "gripper.timestamps", ts, problems)


def check_sync_stage(loaded, problems, stats):
    try:
        image_obj = loaded["image.pkl"]
        tactile_obj = loaded["tactile.pkl"]
        state_obj = loaded["state.pkl"]
        gripper_obj = loaded["gripper.pkl"]

        camera1_ts = np.array(image_obj["camera1"]["timestamps"])
        camera2_ts = np.array(image_obj["camera2"]["timestamps"])
        tactile1_ts = np.array(tactile_obj["tactile1"]["timestamps"])
        tactile2_ts = np.array(tactile_obj["tactile2"]["timestamps"])
        robot_ts = np.array(state_obj["timestamps"])
        gripper_ts = np.array(gripper_obj["timestamps"])

        required = {
            "camera1": camera1_ts,
            "camera2": camera2_ts,
            "tactile1": tactile1_ts,
            "tactile2": tactile2_ts,
            "robot": robot_ts,
            "gripper": gripper_ts,
        }

        for k, arr in required.items():
            if len(arr) == 0:
                problems.append(f"sync stage: {k} timestamps empty before sync")
                return

        sync_info = sync_and_truncate_timestamps_check(required)
        stats["sync_begin"] = sync_info["begin"]
        stats["sync_end"] = sync_info["end"]
        stats["sync_processed_lens"] = sync_info["processed_lens"]

        for k, v in sync_info["processed_lens"].items():
            if v <= 0:
                problems.append(f"sync stage: processed length of {k} is {v}")

    except Exception as e:
        problems.append(f"sync stage failed: {repr(e)}")


def check_episode(ep_dir):
    result = {
        "episode_dir": ep_dir,
        "ok": True,
        "problems": [],
        "stats": {},
        "loaded_files": {},
    }

    problems = result["problems"]
    stats = result["stats"]
    loaded = {}

    # existence
    for fn in REQUIRED_FILES:
        path = os.path.join(ep_dir, fn)
        if not os.path.exists(path):
            problems.append(f"missing required file: {fn}")

    if problems:
        result["ok"] = False
        return result

    # load required
    for fn in REQUIRED_FILES:
        path = os.path.join(ep_dir, fn)
        ok, obj, err = safe_load_pickle(path)
        if not ok:
            problems.append(f"{fn}: failed to load pickle: {err}")
            continue
        loaded[fn] = obj
        result["loaded_files"][fn] = True

    if "image.pkl" in loaded:
        check_image_pkl(loaded["image.pkl"], problems, stats)
    if "tactile.pkl" in loaded:
        check_tactile_pkl(loaded["tactile.pkl"], problems, stats)
    if "state.pkl" in loaded:
        check_state_pkl(loaded["state.pkl"], problems, stats)
    if "gripper.pkl" in loaded:
        check_gripper_pkl(loaded["gripper.pkl"], problems, stats)

    if len(problems) == 0:
        check_sync_stage(loaded, problems, stats)

    result["ok"] = len(problems) == 0
    return result


def find_episode_dirs(root_path):
    dirs = []
    for name in sorted(os.listdir(root_path)):
        full = os.path.join(root_path, name)
        if os.path.isdir(full):
            dirs.append(full)
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, help="dataset root, e.g. /path/to/jishiqi_0316_60")
    parser.add_argument("--episode_length", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    episode_dirs = find_episode_dirs(args.root_path)
    if args.episode_length != -1:
        episode_dirs = episode_dirs[:args.episode_length]

    if len(episode_dirs) == 0:
        print("No episode dirs found.")
        return

    all_results = []
    bad_count = 0

    for ep_dir in episode_dirs:
        result = check_episode(ep_dir)
        all_results.append(result)

        if result["ok"]:
            print(f"[OK]   {ep_dir}")
            if args.verbose:
                pprint(result["stats"])
        else:
            bad_count += 1
            print(f"[BAD]  {ep_dir}")
            for p in result["problems"]:
                print("   -", p)
            if args.verbose and len(result["stats"]) > 0:
                pprint(result["stats"])

    print("\n========== SUMMARY ==========")
    print(f"total episodes: {len(all_results)}")
    print(f"bad episodes:   {bad_count}")
    print(f"good episodes:  {len(all_results) - bad_count}")

    if bad_count > 0:
        print("\nBad episode list:")
        for r in all_results:
            if not r["ok"]:
                print(" -", r["episode_dir"])


if __name__ == "__main__":
    main()