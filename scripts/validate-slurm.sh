#!/usr/bin/env bash
#SBATCH --job-name=actsteer-validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=actsteer-validation-%j.out
set -euo pipefail
# Submit from the clean checkout with its Python environment already activated.
# Override H100 with: sbatch --gres=gpu:nvidia_h200:1 scripts/validate-slurm.sh
: "${ACTSTEER_SHA:?Export the exact candidate commit SHA}"
: "${ACTSTEER_RESULTS:?Export an absolute NEW output directory on persistent storage}"
: "${HF_HOME:?Export the populated Hugging Face cache directory}"
[[ "$ACTSTEER_RESULTS" = /* ]] || { echo 'ACTSTEER_RESULTS must be absolute'; exit 2; }
[[ "$(git rev-parse HEAD)" = "$ACTSTEER_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir "$ACTSTEER_RESULTS"
python -m pip freeze > "$ACTSTEER_RESULTS/pip-freeze.txt"
nvidia-smi > "$ACTSTEER_RESULTS/nvidia-smi.txt"
status=0
for family in gemma llama qwen; do
    python -m actsteer.validation.smoke --family "$family" \
        --manifest configs/validation-models.json \
        --output "$ACTSTEER_RESULTS/$family" || status=1
done
python -m actsteer.validation.accept --sha "$ACTSTEER_SHA" \
    --manifest configs/validation-models.json --output "$ACTSTEER_RESULTS/acceptance.json" \
    "$ACTSTEER_RESULTS/gemma/report.json" "$ACTSTEER_RESULTS/llama/report.json" \
    "$ACTSTEER_RESULTS/qwen/report.json" || status=1
exit "$status"
