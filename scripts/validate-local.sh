#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with its Python environment activated.
: "${1:?Usage: bash scripts/validate-local.sh NEW_OUTPUT_DIRECTORY}"
result="$1"
mkdir -p "$(dirname "$result")"
mkdir "$result"
python -m pip freeze > "$result/pip-freeze.txt"
python -m unittest discover -s tests -v > "$result/unittest.log" 2>&1
python -m ruff check src tests > "$result/ruff.log" 2>&1
python -m pyright --pythonpath "$(command -v python)" > "$result/pyright.log" 2>&1
python -m actsteer.cli config validate examples/configs/smoke.yaml > "$result/cli.log" 2>&1
bash -n scripts/validate-slurm.sh
for family in gemma llama qwen; do
    for mode in chat text; do
        python -m actsteer.validation.smoke --fixture --family "$family" --mode "$mode" \
            --output "$result/$family-$mode" > "$result/$family-$mode.log" 2>&1
    done
done
# Fixture reports must fail the real-checkpoint gate.
if python -m actsteer.validation.accept --sha "$(git rev-parse HEAD)" \
    --output "$result/fixture-rejection.json" \
    "$result/gemma-chat/report.json" "$result/llama-chat/report.json" \
    "$result/qwen-chat/report.json" > "$result/acceptance-negative.log" 2>&1; then
    echo 'ERROR: fixture evidence incorrectly accepted'; exit 1
fi
python - "$result/fixture-rejection.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
assert result['status'] == 'not_validated'
assert 'real-checkpoint' in result['error']
PY
printf 'PASS: local validation only; reports in %s\n' "$result"
