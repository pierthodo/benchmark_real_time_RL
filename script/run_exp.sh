#!/bin/bash

for env in Hopper-v1 Pendulum-v1 Humanoid-v1 HalfCheetah-v1
do
  for algo in PPO SAC ARS
  do
    rllib train --run $algo --env $env --config '{"framework": "torch"}' --checkpoint-freq 25 &
  done
done
wait 
