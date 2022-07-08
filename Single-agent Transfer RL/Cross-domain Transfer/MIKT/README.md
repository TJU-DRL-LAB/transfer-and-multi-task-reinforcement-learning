# Mutual Information Based Knowledge Transfer Under State-Action Dimension Mismatch

arxiv link: https://arxiv.org/abs/2006.07041

by Michael Wan, Tanmay Gangwani, and Jian Peng

> Deep reinforcement learning (RL) algorithms have achieved great success on a wide variety of sequential decision-making tasks. However, many of these algorithms suffer from high sample complexity when learning from scratch using environmental rewards, due to issues such as credit-assignment and high-variance gradients, among others. Transfer learning, in which knowledge gained on a source task is applied to more efficiently learn a different but related target task, is a promising approach to improve the sample complexity in RL. Prior work has considered using pre-trained teacher policies to enhance the learning of the student policy, albeit with the constraint that the teacher and the student MDPs share the state-space or the action-space. In this paper, we propose a new framework for transfer learning where the teacher and the student can have arbitrarily different state- and action-spaces. To handle this mismatch, we produce embeddings which can systematically extract knowledge from the teacher policy and value networks, and blend it into the student networks. To train the embeddings, we use a task-aligned loss and show that the representations could be enriched further by adding a mutual information loss. Using a set of challenging simulated robotic locomotion tasks involving many-legged centipedes, we demonstrate successful transfer learning in situations when the teacher and student have different state- and action-spaces.

This repository is based on [OpenAI Baselines](https://github.com/openai/baselines).

## Installation

### Install MuJoCo

1. Obtain a 30-day free trial or license on the [MuJoCo website](https://www.roboti.us/license.html). You will be emailed a file called `mjkey.txt`.
2. Download the MuJoCo version 1.5 binaries.
3. Unzip the downloaded `mjpro150` directory into `~/.mujoco/mjpro150`,
   and place `mjkey.txt` at `~/.mujoco/mjkey.txt`.
4. ``
pip install -U 'mujoco-py<1.50.2,>=1.50.1'
``

### Install everything else

``
gym==0.10.5
scipy==1.3.3
joblib==0.14.0
cloudpickle==1.2.2
click==7.0
opencv-python==4.1.2.30
tensorflow==1.12.0
num2words==0.5.10
``

This code was tested using Python version 3.6.8.

## Usage
``
python -m baselines.run --env=<name of environment> --source_env=<name of source environment> --alg=ppo2 --num_timesteps=<number of timesteps> --pi_scope=<variable scope for policy> --vf_scope=<variable scope for value function> --network=mlp --teacher_network=mlp --save_path=<directory to save trained models> --save_interval=<save interval> --log_path=<directory to log results> --teacher_paths <file to load teacher weights from> --mapping=learned
``

### Example: Transferring from CentipedeFour to CentipedeEight
``
python -m baselines.run --env=CentipedeEight-v1 --source_env=CentipedeFour-v1 --alg=ppo2 --num_timesteps=2e6 --pi_scope=student0 --vf_scope=vf_student0 --network=mlp --teacher_network=mlp --save_path=models --save_interval=10 --log_path=logs --teacher_paths teacher_models/centipede_four/0/00970 --mapping=learned
``
