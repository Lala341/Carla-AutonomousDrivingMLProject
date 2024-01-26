#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=18:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --export=none
#SBATCH --job-name=trainingppo
#SBATCH --output=outputTrainingPPOKendall13.txt


#sac_vae_seg_multiply042_ffinal multiply
#sac_vae_seg_multiply042_fffinal kendall
#sac_vae_seg_multiply042_ffffinal distance speed
# Your commands to run on the cluster go below this line
echo "Start job!"

module load python/3.9.10/gcc-11.2.0 
module load singularity/3.8.3/gcc-11.2.0
export TMPDIR=/gpfs/users/forerol/tmp

netstat -tuln
SINGULARITYENV_SDL_VIDEODRIVER=offscreen SINGULARITYENV_SDL_HINT_CUDA_DEVICE=0 SINGULARITYENV_CARLA_WORLD_PORT=2000 SINGULARITYENV_WORLD_PORT=2000 singularity exec --nv -e carla-0.9.11.sif /home/carla/CarlaUE4.sh -opengl -benchmark -fps=30 --gpus all --net=host & 
module load anaconda3/2022.10/gcc-11.2.0 
source activate carla
export PYTHONPATH=$PYTHONPATH:~/CARLA_tutorial/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
cd Carla-ppo
#netstat -tuln
echo "Start train"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_DEVICE_NAME="gpu:0"

#reward_speed_centering_angle_multiply
#reward_centering_steer
#reward_kendall
#python run_eval.py --model_name pretrained_agent --record_to_file "models/results_videos/pretrained_agent_map21.avi" --reward_fn reward_centering_steer_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
#python run_eval.py --model_name sac_vae_seg_multiply042_ffinal --record_to_file "models/results_videos/sac_vae_seg_multiply042_ffinal_map2.avi" --reward_fn reward_speed_centering_angle_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
#python run_eval.py --model_name sac_vae_seg_multiply042_fffinal --record_to_file "models/results_videos/sac_vae_seg_kendall042_fffinal_map2.avi" --reward_fn reward_kendall --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
#python run_eval.py --model_name sac_vae_seg_multiply042_ffffinal --record_to_file "models/results_videos/sac_vae_seg_speed_distance042_ffffinal_map2.avi" --reward_fn reward_speed_distance --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
#python run_eval.py --model_name sac_vae_seg_custom042_final --record_to_file "models/results_videos/sac_vae_seg_custom042_final_map2.avi" --reward_fn reward_centering_steer --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
#python run_eval.py --model_name sac_vae_seg_custom_multiply042_final --record_to_file "models/results_videos/sac_vae_seg_custom_multiply042_final_map2.avi" --reward_fn reward_centering_steer_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 

#python train.py --model_name ppo_vae_seg_custom_final --reward_fn reward_centering_steer --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 750
#python train.py --model_name ppo_vae_seg_custom_multiply_final --reward_fn reward_centering_steer_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 750
#python train.py --model_name ppo_vae_seg_kendall --reward_fn reward_kendall --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 150
#python train.py --model_name ttest_ppo_vae_seg_custom_multiply_final --reward_fn reward_centering_steer_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 2
#python train_sac.py --model_name sac_vae_seg_custom_multiply04 --reward_fn reward_centering_steer_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 150
#python train_sac2.py --model_name sac_vae_seg_custom_multiply042_final --reward_fn reward_centering_steer_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 750
#python train.py --model_name ppo_vae_seg_add4 --reward_fn reward_speed_centering_angle_add2 --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 2500
#python train_sac2.py --model_name sac_vae_seg_add4 --reward_fn reward_speed_centering_angle_add2 --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn --num_episodes 2500
echo "Finish train"
echo "Start eval"
#python run_eval.py --model_name sac_vae_seg_custom_multiply04 --record_to_file "models/results_videos/sac_vae_seg_custom_multiply04_map2.avi" --reward_fn reward_centering_steer --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
#python run_eval.py --model_name sac_vae_seg_custom_multiply042_final --record_to_file "models/results_videos/sac_vae_seg_custom_multiply042_final_map2.avi" --reward_fn reward_centering_steer_multiply --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
python run_eval.py --model_name ppo_vae_seg_add4 --record_to_file "models/results_videos/ppo_vae_seg_add42.avi" --reward_fn reward_speed_centering_angle_add2 --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
python run_eval.py --model_name sac_vae_seg_add4 --record_to_file "models/results_videos/sac_vae_seg_add42.avi" --reward_fn reward_speed_centering_angle_add2 --vae_model vae/models/vae_sem_conv_town02_everything --vae_z_dim 64 --vae_model_type cnn 
echo "Done. Finish job"


 