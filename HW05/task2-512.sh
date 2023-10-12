#!/usr/bin/env bash
#SBATCH --job-name=HW05-task2-512
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=HW05-task2-512.out
#SBATCH --error=HW05-task2-512.err

cd $SLURM_SUBMIT_DIR

for i in {10..25}
do
        n=$[2**$i]
        echo "N = $n"
        ./task2 $n 512
        #compute-sanitizer --tool memcheck ./task2 $n 512
        echo ""
done

