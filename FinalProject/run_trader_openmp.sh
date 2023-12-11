#!/usr/bin/env bash
#SBATCH --job-name=final-trader-openmp
#SBATCH --partition=instruction
#SBATCH --time=0-00:5:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=final-trader-openmp.out
#SBATCH --error=final-trader-openmp.err

./bin/trader_openmp ./nasdaq_50_history.csv

