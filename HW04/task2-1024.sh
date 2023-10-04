#!/usr/bin/env bash
#SBATCH --job-name=HW04-task2-1024
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=HW04-task2-1024.out
#SBATCH --error=HW04-task2-1024.err

cd $SLURM_SUBMIT_DIR

for i in {10..29}
do
        n=$[2**$i]
        echo "N = $n"
        ./task2 $n 128 1024
        echo ""
done
