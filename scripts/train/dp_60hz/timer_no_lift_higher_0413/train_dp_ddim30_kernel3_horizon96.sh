# a#!/bin/bash
# a#!/bin/bash
n_obs_steps=1
horizon=96
num_epochs=450 # 数据规模变大了，因此降低epoch数量来减少训练成本。不过training steps数量依然是增加的
kernel_size=3


CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_wipe_image_gelsight_emb_dp_ablation_ensemble_absolute_without_tactile \
    task.dataset_path=/mnt/data/kywang/4090_env/projects/efficient_robot_sys/data/ckpts/timer_no_lift_higher_0413_60hz/rdp_zarr \
    task.name=dp_ddim30_60hz_obs1_timer_no_lift_higher_0413_horizon${horizon}_60hz \
    logging.mode=online \
    policy.noise_scheduler.num_train_timesteps=30 \
    policy.num_inference_steps=30 \
    policy.kernel_size=${kernel_size} \
    horizon=${horizon} \
    n_action_steps=${horizon} \
    +task.dataset.image_downsample_ratio=1 \
    +policy.image_downsample_ratio=1 \
    n_obs_steps=${n_obs_steps} \
    logging.project="diffusion_policy_rdp_for_sys" \
    training.num_epochs=${num_epochs} 

    #training.num_epochs=1000 \
    #dataloader.batch_size=1 \
    #dataloader.num_workers=1 \
    #val_dataloader.batch_size=1 \
    #val_dataloader.num_workers=1