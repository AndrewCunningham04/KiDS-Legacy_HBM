#!/bin/bash
#SBATCH --job-name=create_Q
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=10G
#SBATCH --output="Sampling/Outputs/Q_array%j.out"
#SBATCH --error="Sampling/Outputs/Q_array%j.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2346228@ed.ac.uk

source /home/u6dl/andrew.u6dl/.venv/bin/activate

python "Sampling/Python_files/create_Q.py"