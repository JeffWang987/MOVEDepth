# CUDA_VISIBLE_DEVICES=4 \
DATA_PATH="/mnt/cfs/algorithm/public_data/kitti_depth/kitti_raw"
exp=$1
model_name=$2
PY_ARGS=${@:3}

EXP_DIR=./movedepth/log/$exp
LOG_DIR=$EXP_DIR/$model_name

python -m movedepth.evaluate_depth \
    --data_path $DATA_PATH \
    --dataset kitti \
    --load_weights_folder $EXP_DIR/$model_name"/models/last" \
    --png \
    --height 192 \
    --width 640 \
    --prior_scale 2 \
    --batch_size 1 \
    --eval_split eigen \
    --convex_up \
    --log_dir $EXP_DIR/$model_name \
    $PY_ARGS | tee -a $EXP_DIR/$model_name/log_test.txt
