#!/bin/bash

# Abstraction Learning Ablation Study V2
# Focus: Stable heuristic rollout with systematic hyperparameter sweeps.
# All experiments are run sequentially to ensure stability.

# --- Base Configuration ---
# Define common parameters here to keep individual commands clean.
BASE_ARGS="--causal_rollout False --l 1 --train_iterations 2000 --val_iterations 100 --learning_rate 1e-3 --log_interval 100"

echo "Starting Abstraction Learning Ablation Study (Sequential)..."


# =============================================================================
# EXPERIMENT 1: Baseline (No Abstractions)
# =============================================================================
echo "--- Running baseline (no abstractions) ---"
python -m experiment.train_sorl $BASE_ARGS \
  --run_name "baseline-no_abs" \
  --use_spike_placeholders False \
  --use_rhythmic_placeholders False \
  --abstract_budget 0 \
  --steps 1 \
  --n 2


# =============================================================================
# EXPERIMENT 2: Rhythmic Placeholders Ablation (Varying Steps)
# =============================================================================
echo "--- Testing Rhythmic Placeholders (varying steps) ---"
for steps in 1 3 5; do
  echo "Running Rhythmic with steps=${steps}..."
  python -m experiment.train_sorl $BASE_ARGS \
    --run_name "heuristic-rhythm-s${steps}" \
    --use_spike_placeholders False \
    --use_rhythmic_placeholders True \
    --steps ${steps} \
    --n 2
done


# =============================================================================
# EXPERIMENT 3: Spike Placeholders Ablation (Varying Budget)
# =============================================================================
echo "--- Testing Spike Placeholders (varying abstract budget) ---"
for budget in 2 5 10; do
  echo "Running Spike with budget=${budget}..."
  python -m experiment.train_sorl $BASE_ARGS \
    --run_name "heuristic-spike-b${budget}" \
    --use_spike_placeholders True \
    --use_rhythmic_placeholders False \
    --abstract_budget ${budget} \
    --steps 1 \
    --n 2
done


# =============================================================================
# EXPERIMENT 4: Number of Rollouts (n) Ablation
# Using a good default config (rhythmic + spike)
# =============================================================================
echo "--- Testing Number of Rollouts (n) ---"
for n_rollouts in 2 3 5; do
  echo "Running with n=${n_rollouts}..."
  python -m experiment.train_sorl $BASE_ARGS \
    --run_name "heuristic-both-n${n_rollouts}" \
    --use_spike_placeholders True \
    --use_rhythmic_placeholders True \
    --abstract_budget 5 \
    --steps 3 \
    --n ${n_rollouts}
done


# =============================================================================
# EXPERIMENT 5: Temperature Ablation
# =============================================================================
echo "--- Testing Temperature ---"
for temp in 0.75 1.0 1.5; do
  echo "Running with temperature=${temp}..."
  python -m experiment.train_sorl $BASE_ARGS \
    --run_name "heuristic-both-t${temp}" \
    --use_spike_placeholders True \
    --use_rhythmic_placeholders True \
    --abstract_budget 5 \
    --steps 3 \
    --n 3 \
    --temperature ${temp}
done


# =============================================================================
# EXPERIMENT 6: Memory Fading Ablation
# =============================================================================
echo "--- Testing Memory Fading ---"

echo "Running with Memory Fading ENABLED..."
python -m experiment.train_sorl $BASE_ARGS \
  --run_name "heuristic-both-memfade_ON" \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 3 \
  --n 3 \
  --use_fade_memory True \
  --max_seq_len 1024 \
  --min_keep 32

echo "Running with Memory Fading DISABLED (control)..."
python -m experiment.train_sorl $BASE_ARGS \
  --run_name "heuristic-both-memfade_OFF" \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 3 \
  --n 3 \
  --use_fade_memory False \
  --max_seq_len 1024 \
  --min_keep 32


echo "--- All experiments completed! ---"
echo "Check wandb for results: https://wandb.ai/your-project/abstraction-learning2"

