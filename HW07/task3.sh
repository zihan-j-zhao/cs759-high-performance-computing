#!/usr/bin/env bash
#SBATCH --job-name=HW07-task3
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=HW07-task3.out
#SBATCH --error=HW07-task3.err

cd $SLURM_SUBMIT_DIR

./task3

