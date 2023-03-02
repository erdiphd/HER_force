#!/bin/bash

docker-compose run --rm  -e mujoco_env=FetchPickAndPlace-v1 -e log_tag=log/t1_contact_energy  -e n_epochs=50  -e num_cpu=8 -e  prioritization=contact_energy -e reward_type=sparse her_tactile
