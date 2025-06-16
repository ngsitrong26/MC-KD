#! /bin/bash
set -e
export CUDA_VISIBLE_DEVICES=${1}
MASTER_PORT=${2}
GPUS_PER_NODE=${3}

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
# BASE_PATH=/home/mcn/tue_x/DSKD
BASE_PATH=${4}
CKPT_PATH=${5}
CKPT_SETTING=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-4)"/"$(NF-3)"/"$(NF-2)"/"$(NF-1)}')
MODEL_TYPE=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-6)}')
# MODEL_TYPE="gpt2"

TASK="eval_main"
DATA_NAME=${6}
DATA_DIR="${BASE_PATH}/data/${DATA_NAME}"
DATA_NUM=${9--1}

# hp
VAL_BATCH_SIZE=${7}
SEED=${8}

# runtime
SAVE_PATH=$(dirname ${CKPT_PATH})

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type ${MODEL_TYPE}"
# task
OPTS+=" --task ${TASK}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num ${DATA_NUM}"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/evaluate.py ${OPTS}"
echo ${CMD}

${CMD}
