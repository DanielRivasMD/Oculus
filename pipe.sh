#!/usr/bin/env bash

set -euo pipefail

MODERN="data/cache/fasta/simulation_modern_100nt.fasta"
ANCIENT="data/cache/fasta/simulation_ancient_100nt.fasta"
OUT_DIR="data/cache"

echo "=== Deamination Analysis ==="
julia --project bin/deamination.jl \
    --ancient "$ANCIENT" \
    --modern "$MODERN" \
    --csv "$OUT_DIR/fasta/deamination.csv" \
    --png "data/graph/png/deamination.png"

echo "=== Feature Engineering (minimal) ==="
julia --project bin/features.jl \
    --ancient "$ANCIENT" \
    --modern "$MODERN" \
    --out "$OUT_DIR/features/engineered.csv"

echo "=== Feature Engineering (one-hot) ==="
julia --project bin/features.jl \
    --ancient "$ANCIENT" \
    --modern "$MODERN" \
    --out "$OUT_DIR/features/onehot.csv" \
    --onehot

echo "=== Logistic Regression ==="
julia --project bin/regression.jl \
    --in "$OUT_DIR/features/engineered.csv" \
    --out "$OUT_DIR/inference/logistic_regression.csv"

echo "=== Lasso Regression ==="
julia --project bin/regression.jl \
    --in "$OUT_DIR/features/engineered.csv" \
    --out "$OUT_DIR/inference/lasso_regression.csv" \
    --reg lasso

echo "=== Decision Tree ==="
julia --project bin/decision_tree.jl \
    --in "$OUT_DIR/features/engineered.csv" \
    --out "$OUT_DIR/inference/decision_tree.csv"

echo "=== Random Forest ==="
julia --project bin/decision_tree.jl \
    --in "$OUT_DIR/features/engineered.csv" \
    --out "$OUT_DIR/inference/random_forest.csv" \
    --model forest

echo "=== XGBoost ==="
julia --project bin/decision_tree.jl \
    --in "$OUT_DIR/features/engineered.csv" \
    --out "$OUT_DIR/inference/xgboost.csv" \
    --model xgboost

echo "All analyses completed successfully."
