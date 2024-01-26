#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpua100
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH --export=none
#SBATCH --job-name=hub1
#SBATCH --output=outputStage3.txt


#in cluster
srun --nodes=1 --time=4:00:00 -p gpu --gres=gpu:2 --mem=50G --export=none --propagate=NONE --pty /bin/bash
SBATCH --output=carla_outputts.txt
module load python/3.9.10/gcc-11.2.0
module load singularity/3.8.3/gcc-11.2.0
export TMPDIR=/gpfs/users/forerol/tmp
SINGULARITYENV_SDL_VIDEODRIVER=offscreen SINGULARITYENV_SDL_HINT_CUDA_DEVICE=0 SINGULARITYENV_CARLA_WORLD_PORT=2000 SINGULARITYENV_WORLD_PORT=2000 singularity exec --nv -e carla-0.9.11.sif /home/carla/CarlaUE4.sh -opengl --gpus all --net=host  &
netstat -tuln
#check is running port 2000
module load anaconda3/2022.10/gcc-11.2.0
source activate carla
export PYTHONPATH=$PYTHONPATH:~/CARLA_tutorial/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
cd CARLA_tutorial
#Carla is imported correctly
python3 -c 'import carla;print("Success")'
jupyter notebook --no-browser --port=8889


#in local
ssh -N -L 8889:localhost:8889 forerol@ruche02.mesocentre.universite-paris-saclay.fr

#in ruche02 (open other terminal) change ruche-gpu05 for the ruche assigned in srun
ssh -N -L 8889:localhost:8889 forerol@ruche-gpu05



 