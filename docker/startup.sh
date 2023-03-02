#!/bin/bash

source /home/user/conda/bin/activate her
cd home/user/her

pip install -e /home/user/her/mujoco-py
pip install -e /home/user/her/Gym

cd /home/user/her/Algorithm
sudo chown -R user:user /home/user/her/

python baselines/her/experiment/train.py --env_name  ${mujoco_env} --logdir=${log_tag} --n_epochs=${n_epochs} --num_cpu=${num_cpu} --prioritization=${prioritization} --reward_type=${reward_type}
