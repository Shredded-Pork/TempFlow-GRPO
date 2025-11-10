# Launch command using torchrun (native PyTorch distributed)
torchrun \
    --nproc_per_node=8 \
    --nnodes=6 \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_qwenimage.py \
    --config config/dgx.py:pickscore_qwenimage