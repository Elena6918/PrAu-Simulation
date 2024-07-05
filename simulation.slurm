#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10240M
#SBATCH -t 12:00:00
#SBATCH --partition=standard
#SBATCH -A uvasrg_paid
#SBATCH --job-name="PrAu-Simulations"
#SBATCH --error="simulation.err"
#SBATCH --output="simulation.output"

# export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1

source venv/bin/activate
./run_simulations.sh
