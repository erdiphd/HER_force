#!/bin/bash

docker-compose run --rm  -e mujoco_env=FetchPickAndPlace-v1 -e log_tag=log/t1_non_cum_force  -e n_epochs=50  -e num_cpu=8 -e  prioritization=force -e reward_type=sparse her_tactile
docker-compose run --rm  -e mujoco_env=FetchPickAndPlace-v1 -e log_tag=log/t2_non_cum_force  -e n_epochs=50  -e num_cpu=8 -e  prioritization=force -e reward_type=sparse her_tactile
docker-compose run --rm  -e mujoco_env=FetchPickAndPlace-v1 -e log_tag=log/t3_non_cum_force  -e n_epochs=50  -e num_cpu=8 -e  prioritization=force -e reward_type=sparse her_tactile
docker-compose run --rm  -e mujoco_env=FetchPickAndPlace-v1 -e log_tag=log/t4_non_cum_force  -e n_epochs=50  -e num_cpu=8 -e  prioritization=force -e reward_type=sparse her_tactile
