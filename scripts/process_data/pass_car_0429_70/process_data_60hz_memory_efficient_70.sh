# end_frame多截取5frame，而vase_sponge_test1多截取20frame。因为vase_sponge_test1最后是抬高，而end_frame根据x轴来判断。所以vase_sponge_test1需要多截取
python scripts/process_data/timer_no_lift_higher/process_data_all_zarr_any_frequency_memory_efficient.py \
    --root_path /home/tars/projects/force_data \
    --save_path /home/tars/projects/visual_tactile_policy/kywang/processed_pass_car_0429_70/processed_60hz \
    --task_list cxfxiaoche_0429_70 \
    --episode_length 70 \
    --target_fps 60 \
    --start_z_diff_thresh 0.05 \
    --end_extra_frames 0 \
    # --save_camera_vis \
    # --episode_length 2
