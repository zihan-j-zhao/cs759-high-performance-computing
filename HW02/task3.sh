#!/usr/bin/env bash
#SBATCH --job-name=HW02-task3
#SBATCH --partition=instruction
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:01:30
#SBATCH --output=HW02-task3.out
#SBATCH --error=HW02-task3.err

cd $SLURM_SUBMIT_DIR

./task3
