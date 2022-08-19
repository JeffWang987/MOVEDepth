# CUDA_VISIBLE_DEVICES=4 \
DATA_PATH="/mnt/cfs/algorithm/public_data/kitti_depth/kitti_raw"
exp=$1
model_name=$2
GPU_NUM=$3
BS=$4
PY_ARGS=${@:5}

EXP_DIR=./movedepth/log/$exp
LOG_DIR=$EXP_DIR/$model_name
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node $GPU_NUM -m movedepth.train \
    --dataset kitti \
    --data_path $DATA_PATH \
    --log_dir $EXP_DIR  \
    --model_name $model_name \
    --split eigen_zhou \
    --height 192 \
    --width 640 \
    --prior_scale 2 \
    --png \
    --ddp \
    --batch_size $BS \
    --convex_up \
    --num_workers 12 \
    --learning_rate 2e-4 \
    $PY_ARGS | tee -a $EXP_DIR/$model_name/log_train.txt
