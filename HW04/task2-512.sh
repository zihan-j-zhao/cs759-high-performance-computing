#!/usr/bin/env bash
#SBATCH --job-name=HW04-task2-512
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=HW04-task2-512.out
#SBATCH --error=HW04-task2-512.err

cd $SLURM_SUBMIT_DIR

for i in {10..29}
do
        n=$[2**$i]
        echo "N = $n"
        ./task2 $n 128 512
        echo ""
done
