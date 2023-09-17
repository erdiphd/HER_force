#!/bin/bash
docker-compose run --rm -d -e mujoco_env=FetchPickAndPlace-v1 -e log_tag=log/pickcontact_energy0  -e n_epochs=50  -e num_cpu=8 -e  prioritization=contact_energy -e reward_type=sparse her_tactile
docker-compose run --rm    -e mujoco_env=FetchPickAndPlace-v1 -e log_tag=log/pickcontact_energy1  -e n_epochs=50  -e num_cpu=8 -e  prioritization=contact_energy -e reward_type=sparse her_tactile
