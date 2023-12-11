#!/usr/bin/env bash
#SBATCH --job-name=final-trader
#SBATCH --partition=instruction
#SBATCH --time=0-00:5:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=final-trader.out
#SBATCH --error=final-trader.err

./bin/trader ./nasdaq_100_history.csv

