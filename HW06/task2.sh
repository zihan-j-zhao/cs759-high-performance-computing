#!/usr/bin/env bash
#SBATCH --job-name=HW06-task2
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=HW06-task2.out
#SBATCH --error=HW06-task2.err

cd $SLURM_SUBMIT_DIR

for i in {10..16}
do
        n=$[2**$i]
        echo "N = $n"
        ./task2 $n 1024
        #compute-sanitizer --tool memcheck ./task2 $n 1024
        #cuda-memcheck ./task2 $n 1024
        echo ""
done

