#!/bin/bash$
#SBATCH --job-name=tomopanda-pick_2d_projection$po
#SBATCH --output=job_%j.log$
#SBATCH --error=job_%j.err$
#SBATCH --nodes=1$
#SBATCH --ntasks=1$
#SBATCH --cpus-per-task=4$
#SBATCH --mem=32G$
#SBATCH --partition=cpu$
#SBATCH --time=72:00:00$

# 1) Activate conda environment$
eval "$('/fast/AG_Kudryashev/PatrickMuWang/opt/manba/miniforge3/bin/conda' shell.bash hook)"$
conda activate tomopanda

config_file="$PWD/config.yaml"
cd /fast/AG_Kudryashev/MuWang/github/TomoPANDA-pick/
echo "Running: python utils/2d_projection/main.py -i $config_file"

python utils/2d_projection/main.py -i $config_file