#!/bin/bash

for env in Hopper-v2 Pendulum-v0 Humanoid-v2 HalfCheetah-v2
do
  for algo in PPO SAC ARS
  do
    rllib train --run $algo --env $env --config '{"framework": "torch"}' --checkpoint-freq 25 &
  done
done
wait 
