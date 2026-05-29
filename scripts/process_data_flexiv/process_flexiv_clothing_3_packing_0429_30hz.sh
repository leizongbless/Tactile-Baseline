#!/usr/bin/env bash
# Flexiv_clothing_3_packing_0429: raw HDF5 quaternion is XYZW (scipy native).
# Output dir is suffixed with "corrected" so the previously generated zarr
# (which assumed a single hard-coded quaternion order) is NOT overwritten.

python scripts/process_data_flexiv/process_data_flexiv_zarr.py \
    --root_path /mnt/data/data/ymm/dataset/Flexiv/release \
    --save_path /mnt/data/kywang/visual_tactile/flexiv_dataset \
    --task_list Flexiv_clothing_3_packing_0429 \
    --raw_quat_order xyzw \
    --dual_arm \
    --with_gripper \
    --target_fps 30 \
    --left_camera_name ldl_hand_fisheye \
    --right_camera_name rdl_hand_fisheye \
    --start_z_diff_thresh 0.001 \
    --end_extra_frames 3 \
    --image_num_workers 8 \
    --stage_in_tmpfs \
    --tmpfs_root /dev/shm/flexiv_zarr \
    --output_suffix corrected \
    # --episode_length 2 # for debug
