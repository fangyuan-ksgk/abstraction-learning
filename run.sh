#!/bin/bash

# Abstraction Learning Ablation Study
# Based on experiment notes: focusing on spike placeholders, rhythmic placeholders, steps, n, and temperature

# Base configuration (your current setup)
# BASE_ARGS="--causal_rollout False --l 1 --train_iterations 2000 --val_iterations 100 --learning_rate 1e-3 --log_interval 100"

echo "Starting Abstraction Learning Ablation Study..."

# =============================================================================
# EXPERIMENT 1: Baseline (no abstractions)
# =============================================================================
echo "Running baseline (no abstractions)..."
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders False \
  --use_rhythmic_placeholders False \
  --abstract_budget 0 \
  --steps 1 \
  --n 2

# =============================================================================
# EXPERIMENT 2: Rhythmic Placeholders Only (varying steps)
# =============================================================================
echo "Testing rhythmic placeholders with different steps..."

# Rhythmic + steps=1
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders False \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 2

# Rhythmic + steps=3 (your current setup)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders False \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 3 \
  --n 2

# Rhythmic + steps=5
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders False \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 5 \
  --n 2

# Wait for rhythmic experiments to complete

# =============================================================================
# EXPERIMENT 3: Spike Placeholders with Different Budgets
# =============================================================================
echo "Testing spike placeholders with different abstract budgets..."

# Spike + budget=2 (conservative)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders False \
  --abstract_budget 2 \
  --steps 1 \
  --n 2

# Spike + budget=5 (moderate)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders False \
  --abstract_budget 5 \
  --steps 1 \
  --n 2

# Spike + budget=10 (aggressive)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders False \
  --abstract_budget 10 \
  --steps 1 \
  --n 2

# Wait for spike budget experiments to complete

# =============================================================================
# EXPERIMENT 4: Combined Spike + Rhythmic (best of both)
# =============================================================================
echo "Testing combined spike + rhythmic placeholders..."

# Combined + budget=5
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 2

# Combined + budget=8
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 8 \
  --steps 1 \
  --n 2


# =============================================================================
# EXPERIMENT 5: Temperature Ablation (using best placeholder config)
# =============================================================================
echo "Testing temperature effects..."

# Low temperature (more deterministic)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 0.5 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 2

# Medium temperature (your current)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 2

# High temperature (more exploration)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 2.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 2



# =============================================================================
# EXPERIMENT 6: Number of Rollouts (n) Ablation
# =============================================================================
echo "Testing different numbers of rollouts..."

# Few rollouts (faster training)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 2

# Your current setup
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 3

# More rollouts (better search but slower)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 1 \
  --n 5


# =============================================================================
# EXPERIMENT 7: One Causal Rollout Comparison
# =============================================================================
echo "Running one causal rollout experiment for comparison..."

python -m experiment.train_sorl $BASE_ARGS \
  --temperature 1.0 \
  --causal_rollout True \
  --budget 3 \
  --n 3


# =============================================================================
# EXPERIMENT 8: Curriculum Learning (Temperature Scheduling)
# =============================================================================
echo "Testing temperature curriculum (if implemented)..."

# Start high, decay to low (exploration -> exploitation)
python -m experiment.train_sorl $BASE_ARGS \
  --temperature 2.0 \
  --use_spike_placeholders True \
  --use_rhythmic_placeholders True \
  --abstract_budget 5 \
  --steps 3 \
  --n 3 



echo "All experiments completed!"
echo ""
echo "Experiment Summary:"
echo "1. Baseline (no abstractions)"
echo "2. Rhythmic placeholders with steps={1,3,5}"
echo "3. Spike placeholders with budget={2,5,10}"
echo "4. Combined placeholders with budget={5,8}"
echo "5. Temperature ablation {0.5,1.0,2.0}"
echo "6. Rollout number ablation n={2,3,5}"
echo "7. Single causal rollout comparison"
echo "8. Temperature curriculum"
echo ""
echo "Check wandb for results: https://wandb.ai/your-project/abstraction-learning2"

