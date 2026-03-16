# end_frame多截取5frame，而vase_sponge_test1多截取20frame。因为vase_sponge_test1最后是抬高，而end_frame根据x轴来判断。所以vase_sponge_test1需要多截取
python scripts/process_data/timer/process_data_all_zarr_any_frequency.py \
    --root_path /home/tars/projects/force_data \
    --save_path /home/tars/projects/visual_tactile_policy/kywang/processed_timer/processed_60hz \
    --task_list jishiqi_0316_60 \
    --episode_length 2 \
    --target_fps 30 \
    --start_z_diff_thresh 1.5 \
    --end_extra_frames 3
    # --save_camera_vis \
    # --episode_length 2

# /home/tars/projects/force_data/jishiqi_0316_60