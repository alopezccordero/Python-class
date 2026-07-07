#!/bin/bash
#SBATCH --job-name=humanoid_dir
#SBATCH --partition=gpua30q
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=humanoid_dir_%j.out
#SBATCH --error=humanoid_dir_%j.err

source ~/miniconda3/bin/activate
conda activate mujoco-env

cd ~/Python-class/mujoco/humanoid_direction

echo "Running on $(hostname)"
echo "current directory: #(pwd)"
python --version

python humanoid_direction_train.py

