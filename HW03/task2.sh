#!/usr/bin/env bash
#SBATCH --job-name=HW03-task2
#SBATCH --partition=instruction
#SBATCH --time=0-00:00:30
#SBATCH --gres=gpu:1
#SBATCH --output=HW03-task2.out
#SBATCH --error=HW03-task2.err

cd $SLURM_SUBMIT_DIR

./task2
