#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=6:30:00
#SBATCH -o output_%A.out

cd "$(dirname "$0")"

nvidia-smi
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/tmpl_agent.sqsh

# Run the mess+ evaluation
enroot start \
    --root \
    -e HF_TOKEN_PATH=/dss/dsshome1/09/ge56heh2/.cache/huggingface/token \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf/misc \
    -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets \
    -e HF_DATASETS_TRUST_REMOTE_CODE=True \
    -e VLLM_USE_V1=0 \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/ \
    tmpl_agent \
    /bin/bash vllm_base.sh \
    Qwen/Qwen3-8B-FP8 \
    --num-trials 1 \
    --max-concurrency 20 \
    --vllm-args "--tensor-parallel-size 1 --quantization bitsandbytes --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes  --enable-reasoning --reasoning-parser deepseek_r1"