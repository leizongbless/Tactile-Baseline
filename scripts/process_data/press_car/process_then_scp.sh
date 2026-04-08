# echo "process 15hz"
# bash scripts/process_data/press_car_0407/process_data_30hz.sh

echo "process 60hz"
# bash scripts/process_data/press_car_0407/process_data_60hz.sh

rsync -avzP -e "ssh -p 22" \
    /home/tars/projects/visual_tactile_policy/kywang/processed_press_car_0407_250 \
    root@10.10.54.14:/mnt/data/kywang/visual_tactile/dataset/