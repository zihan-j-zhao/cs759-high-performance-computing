#!/usr/bin/env bash
#SBATCH --job-name=HW06-task1
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=HW06-task1.out
#SBATCH --error=HW06-task1.err

cd $SLURM_SUBMIT_DIR

for i in {5..11}
do
        n=$[2**$i]
        echo "N = $n"
        ./task1 $n 10
        #compute-sanitizer --tool memcheck ./task1 $n 10
        echo ""
done

