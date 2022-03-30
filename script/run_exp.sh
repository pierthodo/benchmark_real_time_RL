#!/bin/bash

for env in Hopper-v2 Pendulum-v0 Humanoid-v2 HalfCheetah-v2 FetchPickAndPlace-v1 HandManipulateBlock-v0
do
  for algo in PPO SAC ARS
  do
    echo "Running $algo $env"
    rllib train --run $algo --env $env --config '{"framework": "torch"}' --checkpoint-freq 100 &
    sleep 30
  done
done
wait 
