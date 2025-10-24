#!/bin/bash

set -e

get_script_dir() {
    local source="${BASH_SOURCE[0]:-${(%):-%x}}"
    cd "$(dirname "$source")" && pwd
}

SCRIPT_DIR="$(get_script_dir)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Root directory: $PARENT_DIR"

# Default values
DOMAIN="airline"
NUM_TRIALS=1
MAX_CONCURRENCY=10
TAU2_PATH="$PARENT_DIR/.venv/bin/tau2"
HOSTED_VLLM_API_BASE="http://0.0.0.0:8000/v1"
TAU2_DOMAINS=("airline" "retail" "telecom")
VLLM_API_KEY="my-api-key"

# Check if required arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model-name> [options]"
    echo "Options:"
    echo "  --num-trials <n>            Number of trials (default: 1)"
    echo "  --max-concurrency <n>       Max concurrency (default: 10)"
    echo "  --vllm-args '<args>'        Additional vLLM arguments"
    exit 1
fi

MODEL_NAME=$1
shift

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --vllm-args)
            VLLM_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cleanup() {
    if [ ! -z "$VLLM_PID" ]; then
        echo "Shutting down vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        echo "vLLM server stopped."
    fi
}

trap cleanup EXIT INT TERM

echo "Starting vLLM server with model: $MODEL_NAME"
TRANSFORMERS_OFFLINE=1 $PARENT_DIR/.venv/bin/vllm serve $MODEL_NAME --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" $VLLM_ARGS &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

max_attempts=180
sleep_time=2
echo "Waiting for vLLM server to be ready ("$(($max_attempts * $sleep_time))" seconds)..."
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    attempt=$((attempt + 1))
    sleep $sleep_time
done

if [ $attempt -eq $max_attempts ]; then
    echo "Error: vLLM server failed to start within timeout"
    exit 1
fi

export HOSTED_VLLM_API_KEY="$VLLM_API_KEY"
for DOMAIN in "${TAU2_DOMAINS[@]}"; do
    echo "Running tau2-bench..."
    echo "  Domain: $DOMAIN"
    echo "  Trials: $NUM_TRIALS"
    echo "  Max Concurrency: $MAX_CONCURRENCY"



    $PARENT_DIR/.venv/bin/tau2 run \
        --domain $DOMAIN \
        --agent-llm hosted_vllm/$MODEL_NAME \
        --user-llm gpt-4o-mini \
        --num-trials $NUM_TRIALS \
        --max-concurrency $MAX_CONCURRENCY

    echo "$DOMAIN done."

done

echo "tau2-bench run completed successfully."
