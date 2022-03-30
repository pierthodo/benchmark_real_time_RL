#!/bin/bash

for env in Hopper-v2 Pendulum-v0 Humanoid-v2 HalfCheetah-v2 FetchPickAndPlace-v1 HandManipulateBlock-v0
do
  for algo in PPO SAC ARS
  do
    rllib train --run $algo --env $env --config '{"framework": "torch"}' --checkpoint-freq 100 &
  done
done
wait 
