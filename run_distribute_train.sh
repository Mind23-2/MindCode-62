#!/bin/bash
if [ $# != 1 ]
then
    echo "Usage: sh run_distribute_train.sh [MINDSPORE_HCCL_CONFIG_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

#DATASET_PATH=$(get_real_path $1)
#PRETRAINED_BACKBONE=$(get_real_path $2)
MINDSPORE_HCCL_CONFIG_PATH=$(get_real_path $1)
#echo $DATASET_PATH
#echo $PRETRAINED_BACKBONE
echo $MINDSPORE_HCCL_CONFIG_PATH

#if [ ! -d $DATASET_PATH ]
#then
#    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
#exit 1
#fi

#if [ ! -f $PRETRAINED_BACKBONE ]
#then
#    echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
#exit 1
#fi

#if [ ! -f $MINDSPORE_HCCL_CONFIG_PATH ]
#then
#    echo "error: MINDSPORE_HCCL_CONFIG_PATH=$MINDSPORE_HCCL_CONFIG_PATH is not a file"
#exit 1
#fi

export DEVICE_NUM=8
export RANK_SIZE=4
export MINDSPORE_HCCL_CONFIG_PATH=$MINDSPORE_HCCL_CONFIG_PATH
export IS_DISTRIBUTED=True
device_id=(4 5 6 7)
ulimit -n 10240
for((j=0; j<$RANK_SIZE; j++))
do
    export DEVICE_ID=${device_id[$j]}
    export RANK_ID=${j}
    rm -rf ./train_parallel${device_id[$j]}
    mkdir ./train_parallel${device_id[$j]}
    cp ./*.py ./train_parallel${device_id[$j]}
    cp -r ./datalist ./train_parallel${device_id[$j]}
    cd ./train_parallel${device_id[$j]} || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python r2plus1d_train.py \
        > log.txt 2>&1 &
    cd ..
done
