#!/usr/bin/env bash
#SBATCH --job-name=final-trader-cuda
#SBATCH --partition=instruction
#SBATCH --time=0-00:5:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=final-trader-cuda.out
#SBATCH --error=final-trader-cuda.err

./bin/trader_cuda ./nasdaq_50_history.csv

