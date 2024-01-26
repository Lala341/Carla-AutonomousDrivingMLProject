#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --export=none
#SBATCH --job-name=hub1
#SBATCH --output=outputJob.txt



# Your commands to run on the cluster go below this line
echo "Hello, this is my Slurm job!"


module load python/3.9.10/gcc-11.2.0
module load singularity/3.8.3/gcc-11.2.0

singularity cache clean
export TMPDIR=/gpfs/users/forerol/tmp
# Run your Python script
singularity build carla-0.9.11.sif docker://carlasim/carla:0.9.11



 