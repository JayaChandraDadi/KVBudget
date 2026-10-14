#!/usr/bin/env bash
# Toy-scale experiments (any machine; needs `pip install "jax[cpu]" optax` for training).
set -euo pipefail
cd "$(dirname "$0")/../examples"
python toy_experiments.py --out ../results/toy "$@"
