#!/usr/bin/env bash
#SBATCH --job-name=HW03-task3
#SBATCH --partition=instruction
#SBATCH --time=0-00:01:30
#SBATCH --gres=gpu:1
#SBATCH --output=HW03-task3.out
#SBATCH --error=HW03-task3.err

cd $SLURM_SUBMIT_DIR

for i in {10..29}
do
        n=$[2**$i]
        echo "N = $n"
        ./task3 $n
        echo ""
done
