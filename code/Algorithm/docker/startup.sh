#!/bin/bash
#docker container run --rm  -e mujoco_env="FetchPickAndPlace-v1" -e learn_alg="normal" -e buffer_type="energy" -e log_tag="energy1" -v /home/nrp/Desktop/hgg_poet1:/home/user/hgg_test --name hgg_docker_work_instance_1 -it erdiphd/hgg_test

# sudo /usr/bin/supervisord -c /etc/supervisor/supervisord.conf  > /dev/null 2>&1 &
# /home/user/.conda/envs/HGG/bin/python home/user/HGG/train.py
# /usr/bin/bash
source /home/user/conda/bin/activate her
cd home/user/her

pip install -e /home/user/her/mujoco-py
pip install -e /home/user/her/Gym

cd /home/user/her/Algorithm
sudo chown -R user:user /home/user/her/

python baselines/her/experiment/train.py --env_name  ${mujoco_env} --logdir=${log_tag} --n_epochs=${n_epochs} --num_cpu=${num_cpu} --prioritization=${prioritization} --reward_type=${reward_type}

# docker-compose run -e mujoco_env=FetchPickAndPlace-v1 -e learn_alg=her -e log_tag=log/r1 -e num_env=2 -e num_timesteps=1e6 her
# /bin/bashx