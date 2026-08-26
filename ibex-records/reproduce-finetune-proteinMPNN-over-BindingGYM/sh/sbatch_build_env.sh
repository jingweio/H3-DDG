#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=bgym_env
#SBATCH --output=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/build_env_%j.out
#SBATCH --error=/ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/build_env_%j.err

# Build the env inside a compute job, not on the login node.
#
# The login-node attempt was SIGKILLed part-way through `conda create`
# ("3235159 Killed $CONDA_EXE"). Not memory pressure -- the node had 272 GB free -- so it is a
# per-user cgroup or watchdog limit on login-node processes, which pulling the 1.8 GB
# torch 1.13.1 wheel and unpacking four compiled extensions is exactly the shape of work to trip.
#
# No GPU requested: nothing here needs one, and CPU-only jobs schedule almost immediately.
# Verification does check that torch was built against CUDA, which does not require a device.

set -euo pipefail
bash /ibex/user/guoj0f/H3-DDG/reproduce/ibex-records/reproduce-finetune-proteinMPNN-over-BindingGYM/sh/build_env_bgym_official.sh "${1:---force}"
