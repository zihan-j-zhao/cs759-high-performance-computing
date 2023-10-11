#!/usr/bin/env bash
#SBATCH --job-name=HW05-task1-16
#SBATCH --partition=instruction
#SBATCH --time=0-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=HW05-task1-16.out
#SBATCH --error=HW05-task1-16.err

cd $SLURM_SUBMIT_DIR

for i in {5..14}
do
        n=$[2**$i]
        echo "N = $n"
        ./task1 $n 16
        #compute-sanitizer --tool memcheck ./task1 $n 16
        echo ""
done

