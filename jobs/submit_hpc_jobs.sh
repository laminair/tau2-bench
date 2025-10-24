#!/bin/bash

job_id=$(sbatch --parsable qwen3_0.6b.sbatch)
echo "Job evaluating Qwen 3 0.6B submitted"
sleep 1

job_id=$(sbatch --dependency=afterany:$job_id --parsable qwen3_1.7b.sbatch)
echo "Job evaluating Qwen 3 1.7B submitted"
sleep 1

job_id=$(sbatch --dependency=afterany:$job_id --parsable qwen3_4b.sbatch)
echo "Job evaluating Qwen 3 4B submitted"
sleep 1

job_id=$(sbatch --dependency=afterany:$job_id --parsable qwen3_8b.sbatch)
echo "Job evaluating Qwen 3 8B submitted"
sleep 1

job_id=$(sbatch --dependency=afterany:$job_id --parsable qwen3_14b.sbatch)
echo "Job evaluating Qwen 3 14B submitted"
sleep 1

job_id=$(sbatch --dependency=afterany:$job_id --parsable qwen3_32b.sbatch)
echo "Job evaluating Qwen 3 32B submitted"
sleep 1

job_id=$(sbatch --dependency=afterany:$job_id --parsable qwen3_30b3a.sbatch)
echo "Job evaluating Qwen 3 30B-A3B submitted"
sleep 1
