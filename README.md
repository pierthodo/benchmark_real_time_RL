# Benchmark real-time reinforcement learning

This is a repository containing the dockerfile used to train and evaluate the algorithms for the real-time continuous control benchmark. 

The docker container can simulate embedded device by restricting access to cpu cycle following https://docs.docker.com/config/containers/resource_constraints/

The dockerfile pulls a forked version of the repository containing the RL algorithms. The codebase used to train the algorithm were used from either well-tested repository or published by the authors themselves.

ARS: https://github.com/ray-project/ray \\
RTRL: https://github.com/rmst/rtrl
SAC, PPO: https://github.com/openai/spinningup
MBPO, PETS, PLANET: https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl
DREAMER: https://github.com/danijar/dreamerv2

All the hyper-parameters used were the one recommended by the the libraries. The training performance can be visualized in a wandb dashboard https://wandb.ai/pierthodo/RTDM_train, the inference time https://wandb.ai/pierthodo/RTDM_inference and the robustness to real time https://wandb.ai/pierthodo/RTDM_performance. 

The command line used to train the algorithm from the respective libraries. 

## SAC PPO 

```
cd /app/spinningup
python -m spinup.run ppo --env continuous_CartPole-v0 --data_dir /root/logdir/spinup/ppo --epochs 25 --seed 1 2 3 4 5
python -m spinup.run sac --env continuous_CartPole-v0 --data_dir /root/logdir/spinup/continuous_CartPole-v0/sac --epochs 25 --seed 1 2 3 4 5
```


## MBPO PETS PLANET
```
cd /app/mbrl-lib
python -m mbrl.examples.main algorithm=mbpo overrides=mbpo_halfcheetah seed=1 
python -m mbrl.examples.main algorithm=pets overrides=pets_halfcheetah seed=1 
python -m mbrl.examples.main algorithm=planet overrides=planet_cartpole_balance dynamics_model=planet
```

## RTRL
```
cd /app/rtrl
python -m rtrl run-fs exp/HalfCheetah-v2-RTAC rtrl:RtacTraining Env.id=HalfCheetah-v2 Env.real_time=True seed=1
```

## ARS
```
rllib train --run ARS --env Hopper-v2 --config '{"framework": "torch"}' --checkpoint-freq 100 
```

## DREAMER 

Used DockerFile from Dreamerv2 (due to compatibility problem with my docker) https://github.com/danijar/dreamerv2/blob/52fc568f46d25421fbdd4daf75fddd6feabca8d4/Dockerfile
```
python3 dreamerv2/train.py --logdir ~/logdir/dmc_cartpole_balance/dreamerv2/1 --configs dmc_vision --task dmc_cartpole_balance
```

## Evaluation (WORK IN PROGRESS)

The script used to evaluate the performance can be found in scripts/evaluate.py

The discretization time can be modified by changing the timestep in the xml files of the mujoco library directly. 
