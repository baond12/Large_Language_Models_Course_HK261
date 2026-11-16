#!/usr/bin/env bash
# Run inside the Docker container; launch docker exec from a host tmux session.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_dir="$(cd -- "$script_dir/../.." && pwd)"
run_name="${1:-instructor_baseline_full_2745_01}"
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
  printf 'Invalid run name\n' >&2
  exit 2
fi
log_dir="$work_dir/artifacts/notebook_execution_20261116"
mkdir -p -- "$log_dir"
log_file="$log_dir/$run_name.console.log"
if [[ -e "$log_file" ]]; then
  printf 'Console log already exists; use a new run name: %s\n' "$log_file" >&2
  exit 2
fi
cd -- "$work_dir"
printf 'Full RES baseline: 2745 pairs, GPU selection %s\nLog: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}" "$log_file"
python3 -u "$script_dir/run_instructor_notebook.py" \
  --execute \
  --mode notebook \
  --evaluation-set full-res \
  --run-name "$run_name" \
  --reuse-index-from instructor_pilot_1q_01 \
  2>&1 | tee "$log_file"
printf '\nFull RES execution command completed successfully. Review summary.json for QC results.\n'
