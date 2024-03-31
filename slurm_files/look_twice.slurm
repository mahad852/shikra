#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=2 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=13:59:59 # Run for 14 hours at most
#SBATCH --job-name=SHIKRA_LOOK_TWICE  # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user ma649596@ucf.edu # Use this address
#SBATCH --output=outputs/look_twice.out
#SBATCH --error=outputs/look_twice.err
#SBATCH --constraint=gpu80

# Load modules
module load anaconda/anaconda3

source activate shikra

accelerate launch --num_processes 2 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_look_twice.py \
        --cfg-options model_args.model_name_or_path=shikra-7b