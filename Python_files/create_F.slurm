#!/bin/bash
#SBATCH --job-name=norm_512
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output="Sampling/Outputs/F_array%j.out"
#SBATCH --error="Sampling/Outputs/F_array%j.err"

source /home/u6dl/andrew.u6dl/.venv/bin/activate

python "Sampling/Python_files/create_F.py" \
    --num_warmup 500 \
    --num_samples 200 \
    --num_gals 200000 \
    --num_chains 32 \
    --filename "norm_512.npy"

python "Sampling/Python_files/create_F.py" \
    --num_warmup 100 \
    --num_samples 200 \
    --num_gals 200000 \
    --num_chains 32 \
    --filename "norm_512_100.npy"