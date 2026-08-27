#!/bin/bash
#SBATCH --job-name=test_warmup
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output="Sampling/Outputs/F_array%j.out"
#SBATCH --error="Sampling/Outputs/F_array%j.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2346228@ed.ac.uk

source /home/u6dl/andrew.u6dl/.venv/bin/activate

python "Sampling/Python_files/create_F.py" \
    --num_warmup 10 \
    --num_samples 200 \
    --num_gals 10000 \
    --num_chains 32 \
    --filename "test_num_warmup_10.npy"

python "Sampling/Python_files/create_F.py" \
    --num_warmup 50 \
    --num_samples 200 \
    --num_gals 10000 \
    --num_chains 32 \
    --filename "test_num_warmup_50.npy"

python "Sampling/Python_files/create_F.py" \
    --num_warmup 100 \
    --num_samples 200 \
    --num_gals 10000 \
    --num_chains 32 \
    --filename "test_num_warmup_100.npy"

python "Sampling/Python_files/create_F.py" \
    --num_warmup 500 \
    --num_samples 200 \
    --num_gals 10000 \
    --num_chains 32 \
    --filename "test_num_warmup_500.npy"

python "Sampling/Python_files/create_F.py" \
    --num_warmup 1000 \
    --num_samples 200 \
    --num_gals 10000 \
    --num_chains 32 \
    --filename "test_num_warmup_1000.npy"
