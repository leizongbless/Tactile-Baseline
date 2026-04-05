# echo "process 15hz"
# bash scripts/process_data/hit_mouse/process_data_30hz.sh

echo "process 60hz"
# bash scripts/process_data/hit_mouse/process_data_60hz.sh

# scp -P 1030 -r  /home/tars/projects/visual_tactile_policy/kywang/processed_hit_mouse_0402_100 root@8.130.212.67:/mnt/data/kywang/visual_tactile/dataset/
rsync -avzP -e "ssh -p 1030" \
    /home/tars/projects/visual_tactile_policy/kywang/processed_hit_mouse_0402_100 \
    root@8.130.212.67:/mnt/data/kywang/visual_tactile/dataset/