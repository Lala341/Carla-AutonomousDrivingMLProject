# Autonomous Driving Simulation with CARLA and Reinforcement Learning

## Overview

This project simulates autonomous driving in a virtual environment using CARLA as the simulation platform. The primary focus is on exploring different policies for continuous control using reinforcement learning techniques.

- Simulation Platform: [CARLA](http://carla.org/) (version 0.9.11)
- Reinforcement Learning Algorithms: Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC)

## Repository Structure

The project is based on the [CARLA PPO agent](https://github.com/bitsauce/Carla-ppo) as environment of Carla. In this repository, we have implemented and customized our agents, including training SAC models, and introduced various modifications to VAE and Carla compatibility issues different parts of the code.

### Additional Features

- Customized gym-like environments for CARLA:
  1. Lap environment: Focuses on training an agent to follow a predetermined lap.
  2. Route environment: Focuses on training agents that can navigate from point A to point B.

- Analysis of optimal PPO parameters, environment designs, reward functions, etc., to find the optimal setup for training reinforcement learning-based autonomous driving agents.

- Training a Variational Autoencoder (VAE) for improved state representation.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.6
- [CARLA 0.9.11](https://github.com/carla-simulator/carla/tree/0.9.11) (or later)
- TensorFlow for GPU (version 1.13 or later)
- OpenAI gym (version 0.12.0)
- OpenCV for Python (version 4.0.0)
- GPU with at least 4 GB VRAM (tested on GeForce GTX 970)

### Running a Trained Agent

We provide a pretrained PPO agent for the lap environment. To run it:

```bash
python run_eval.py --model_name ppo_vae_seg_add4 -start_carla




