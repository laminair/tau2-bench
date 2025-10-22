#!/bin/bash

bash vllm_base.sh Qwen/Qwen3-14B \
    --num-trials 1 \
    --max-concurrency 10 \
    --vllm-args "--tensor-parallel-size 1 --quantization bitsandbytes --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes  --enable-reasoning --reasoning-parser deepseek_r1"
