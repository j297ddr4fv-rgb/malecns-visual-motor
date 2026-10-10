#!/usr/bin/env bash
# Run one matched seed of the nonvisual receptor experiment matrix.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="/data/artifacts/by-project/malecns-gpu/experiments/receptor-study-20260925"
runtime="/data/environments/projects/flyvis-opticlobe/bin/python"
seed="${1:-1729}"
assay="${2:-all}"

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  printf 'seed must be a nonnegative integer\n' >&2
  exit 2
fi
if [[ "$assay" != all && "$assay" != gust && "$assay" != hand &&
      "$assay" != diagnostic && "$assay" != hand_air && "$assay" != replay ]]; then
  printf 'assay must be all, gust, hand, diagnostic, hand_air, or replay\n' >&2
  exit 2
fi

run_case() {
  local name="$1" config="$2" scene="$3" stimulus="$4" ablation="${5:-}"
  local stimulus_config="${6:-escape_stimuli.json}"
  local out="$artifact_root/${name}-seed${seed}"
  local -a intervention=()
  if [[ -n "$ablation" ]]; then
    local -a groups=()
    IFS=',' read -r -a groups <<< "$ablation"
    for group in "${groups[@]}"; do
      intervention+=(--ablate-mechanosensory "$group")
    done
  fi
  mkdir -p "$out"
  printf 'running %s seed %s\n' "$name" "$seed"
  systemd-run --user --wait --pipe --collect \
    --property=MemoryMax=5G --property=LimitMEMLOCK=64M --property=RuntimeMaxSec=300 \
    --setenv="PYTHONPATH=$project_root/scripts" \
    --setenv=FLYVIS_ROOT_DIR=/data/models/flyvis \
    --setenv=WARP_CACHE_PATH=/data/cache/malecns-gpu/warp \
    --setenv=PYTHONUNBUFFERED=1 \
    "$runtime" "$project_root/scripts/run_digital_fly.py" \
    --mode escape-experiment --strict \
    --config "$project_root/config/$config" \
    --scene-manifest "$project_root/config/$scene" \
    --scenario escape --neural malecns --vision flyvis --body reduced \
    --renderer-backend warp_cuda --neural-device cuda --neural-backend triton \
    --neural-cuda-graph --vision-device cuda:0 --max-vram-gib 15.5 \
    --seed "$seed" --duration 2.5 --telemetry-mode minimal \
    --stimulus-config "$project_root/config/$stimulus_config" \
    --stimulus-name "$stimulus" "${intervention[@]}" --out "$out"
}

if [[ "$assay" == all || "$assay" == gust ]]; then
  run_case gust-calm-intact receptor_experiment_calm.json realistic_scene.json sham_grounded
  run_case gust-positive-intact receptor_experiment_gust.json realistic_scene.json sham_grounded
  run_case gust-calm-jo-off receptor_experiment_calm.json realistic_scene.json sham_grounded jo
  run_case gust-positive-jo-off receptor_experiment_gust.json realistic_scene.json sham_grounded jo
fi
if [[ "$assay" == all || "$assay" == hand ]]; then
  run_case hand-dark-intact free_escape_experiment.json hand_experiment_dark.json looming_grounded
  run_case hand-dark-haltere-off free_escape_experiment.json hand_experiment_dark.json looming_grounded haltere
  run_case hand-dark-wing-off free_escape_experiment.json hand_experiment_dark.json looming_grounded wing
  run_case hand-dark-leg-off free_escape_experiment.json hand_experiment_dark.json looming_grounded leg
fi
if [[ "$assay" == diagnostic ]]; then
  run_case gust-positive-all-mech-off receptor_experiment_gust.json realistic_scene.json sham_grounded jo,haltere,wing,leg
fi
if [[ "$assay" == hand_air ]]; then
  run_case hand-textured-no-air-intact free_escape_experiment.json hand_experiment_textured.json looming_grounded
  run_case hand-textured-air-intact free_escape_experiment.json hand_experiment_textured.json looming_grounded '' receptor_hand_air_stimuli.json
  run_case hand-textured-no-air-jo-off free_escape_experiment.json hand_experiment_textured.json looming_grounded jo
  run_case hand-textured-air-jo-off free_escape_experiment.json hand_experiment_textured.json looming_grounded jo receptor_hand_air_stimuli.json
fi
if [[ "$assay" == replay ]]; then
  run_case gust-positive-intact-repeat receptor_experiment_gust.json realistic_scene.json sham_grounded
  run_case hand-dark-intact-repeat free_escape_experiment.json hand_experiment_dark.json looming_grounded
fi
