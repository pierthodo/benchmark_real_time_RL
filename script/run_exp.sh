#!/bin/bash

for i in {0..9}
do
  rllib train --run PPO --env Hopper-v2 --config '{"framework": "torch"}' --checkpoint-freq 25 &
done
wait 
