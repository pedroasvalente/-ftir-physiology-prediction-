#!/bin/bash
# GCP Compute Engine startup script for training on a Spot VM.
# Attach as instance metadata key "startup-script" when creating the VM.
#
# Required metadata / environment variables:
#   GITHUB_REPO       — full repo URL, e.g. https://github.com/pedroasvalente/ftir-physiology-prediction.git
#   DAGSHUB_USER_TOKEN — DagsHub personal access token
#   CONFIG            — path to experiment config inside the repo (default: experiments/configs/all_targets.json)
#
# Recommended VM: n2-standard-16 Spot (~$0.23/hr)

set -euo pipefail

REPO_URL="${GITHUB_REPO:?GITHUB_REPO metadata is required}"
REPO_DIR="/opt/ftir-physiology-prediction"
VENV_DIR="$REPO_DIR/.venv"
CONFIG="${CONFIG:-experiments/configs/all_targets.json}"

apt-get update -qq && apt-get install -y -qq git python3-venv

git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -e . -q

export DAGSHUB_USER_TOKEN="${DAGSHUB_USER_TOKEN:?DAGSHUB_USER_TOKEN metadata is required}"
export RANDOM_SEED=52
export R2_THRESHOLD=0.3

"$VENV_DIR/bin/ftir-pred" train "$CONFIG"

echo "Training complete."
