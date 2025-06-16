#! /bin/bash
GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
BASE_PATH=/mnt/nam_x/tue_x/DSKD
CKPT_TYPE="gpt2"
CKPT_NAME="gpt2-base"
CKPT_PATH="/mnt/nam_x/tue_x/DSKD/model_hub/${CKPT_TYPE}/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen1.5-1.8B"
TEACHER_MODEL_PATH="/mnt/nam_x/tue_x/DSKD/model_hub/qwen/Qwen1.5-1.8B"
DATA_DIR="${BASE_PATH}/data/dolly/"
TASK="MultiLevelOT"
BATCH_SIZE=4
LR=0.0005
GRAD_ACC=2
EVAL_BATCH_SIZE=4
EPOCH=10
KD_RATE=5.0
KD_TEMP=2.0
MAX_LENGTH=512
PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001
PRECISION="bf16"
CRITERION="MultiLevelOT"
KD_OBJ="forward_kl"
CONFIG="${KD_OBJ}-${PRECISION}"
SETTING="criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=${BATCH_SIZE}*${GRAD_ACC}*${GPUS_PER_NODE}*${NNODES}__lr=${LR}"
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
SEED=10
mkdir -p ${SAVE_PATH}
OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --task ${TASK}"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
# distiller
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --load /mnt/nam_x/tue_x/DSKD/outputs/gpt2/gpt2-base/MultiLevelOT_1/criterion=MultiLevelOT_1__forward_kl-bf16__teacher=Qwen1.5-1.8B__kd^rate=5.0__kd^temp=2.0__epoch=20__bsz=4x2x1=4*2*1*1__lr=0.0005/epoch13_step18577_loss6.0217_rougel24.4114"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"
OPTS+=" --hidden-dim-student 768"
OPTS+=" --hidden-dim-teacher 2048"
OPTS+=" --max-student-len 512"
OPTS+=" --max-teacher-len 512"
OPTS+=" --proj_dim 256"
OPTS+=" --top_k_vocab 300"
OPTS+=" --ot_weight_logits 0.1"
OPTS+=" --ot_weight_hidden 0.1"
OPTS+=" --ce_weight 2.0"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
if [[ "$PRECISION" == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ "$PRECISION" == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ "$PRECISION" == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}"
${CMD}