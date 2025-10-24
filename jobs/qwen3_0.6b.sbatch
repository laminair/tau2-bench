#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH -o output_%A.out

cd "$(dirname "$0")"

nvidia-smi
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/tmpl_agent_v3.sqsh

# Run the mess+ evaluation
enroot start \
    --root \
    -e HF_TOKEN_PATH=/dss/dsshome1/09/ge56heh2/.cache/huggingface/token \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/hf/misc \
    -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/.cache/datasets \
    -e HF_DATASETS_TRUST_REMOTE_CODE=True \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/ \
    tmpl_agent_v3 \
    /bin/bash /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/tau2-bench/jobs/vllm_base.sh \
    Qwen/Qwen3-0.6B \
    --num-trials 1 \
    --max-concurrency 20 \
    --vllm-args "--tensor-parallel-size 1 --quantization bitsandbytes --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes"