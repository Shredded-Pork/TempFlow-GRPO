#!/bin/bash
# Common part for all nodes 
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --num_machines 6 --num_processes 48 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${PORT} \
    scripts/train_flux_pr.py \
    --config config/dgx.py:flux_pr

