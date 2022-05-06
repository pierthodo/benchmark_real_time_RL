import os 
os.environ["MUJOCO_GL"] = "osmesa"

from cmath import inf
import gym
import time
import wandb
# from ray import serve
from pydoc import doc
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ars as ars
import ray.rllib.agents.sac as sac
import math
import rtrl
import os
import time
import copy
from rtrl import Training, run
from rtrl.wrappers import StatsWrapper
import numpy as np
import omegaconf
import torch
import mbrl.util.env
import mbrl.util.common
import mbrl.planning
import mbrl.env.pets_pusher as pusher
import mbrl.env.humanoid_truncated_obs as humanoid
import mbrl.env.cartpole_continuous as cart
import argparse
from mbrl.planning.core import load_agent
from spinup.utils.test_policy import load_policy_and_env, run_policy


def load(path,algo,env_name="Hopper-v2",gpu=False):
    if gpu:
        print("Using GPU")
    else:
        print("Using CPU")
    import numpy as np
    if algo == "mbpo":
        import omegaconf
        import mbrl
        cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
        env, _, _ = mbrl.util.env.EnvHandler.make_env(cfg)

        from mbrl.planning.core import load_agent
        device = "cuda" if gpu else "cpu"
        ag = load_agent(path, env,device)
        agent = lambda obs: ag.act(obs, deterministic=True)
    elif algo == "planet":
        import omegaconf
        import torch
        import hydra
        from mbrl.models import ModelEnv, ModelTrainer
        from mbrl.env.termination_fns import no_termination
        from mbrl.planning import RandomAgent, create_trajectory_optim_agent_for_model
        from mbrl.util.env import EnvHandler
        import numpy as np
        import os
        os.environ["MUJOCO_GL"] = "osmesa"
        cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
        env, term_fn, reward_fn = EnvHandler.make_env(cfg)
        torch_generator = torch.Generator(device=cfg.device)
        cfg.dynamics_model.action_size = env.action_space.shape[0]
        planet = hydra.utils.instantiate(cfg.dynamics_model)
        planet.load(path)
        device = "cuda" if gpu else "cpu"
        planet.to(device)
        model_env = ModelEnv(env, planet, no_termination, generator=torch_generator)
        ag = create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent)

        def agent(obs,prev_action,done=False):
            import numpy as np
            if done:
                print("reset agent")
                ag.reset()
                planet.reset_posterior()
                planet.update_posterior(obs, action=None, rng=torch_generator)
            else:
                planet.update_posterior(obs, action=prev_action, rng=torch_generator)

            return np.clip(ag.act(obs),-1,1)

    elif algo == "pets":
        import mbrl.util.env
        import mbrl.util.common
        import mbrl.planning
        import omegaconf
        import torch
        cfg = omegaconf.OmegaConf.load(path+".hydra/config.yaml")
        if not gpu:
            cfg["device"] = "cpu"
        torch_generator = torch.Generator(device=cfg.device)
        env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
        dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, env.observation_space.shape, env.action_space.shape)
        dynamics_model.load(path)
        if gpu == False:
            dynamics_model.to("cpu")
        model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
        ag = mbrl.planning.create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles)
        action = ag.act(env.reset())
        action = np.clip(action, -1.0, 1.0)  # to account for the noise
        agent = lambda obs: np.clip(ag.act(obs, deterministic=True),-1.0,1.0)
    elif algo == "rtrl":
        import rtrl
        from rtrl.wrappers import Float64ToFloat32, TimeLimitResetWrapper, NormalizeActionWrapper
        r = rtrl.load(path+"state")
        if not gpu:
            r.agent.model.to("cpu")
        else:
            r.agent.model.to("cuda")
        agent = lambda obs: r.agent.act(obs,[],[],[],train=False)[0]
        env = NormalizeActionWrapper(TimeLimitResetWrapper(Float64ToFloat32(gym.make(env_name))))
    elif algo == "ars":
        checkpoint_num = {"continuous_CartPole-v0":"050","HalfCheetah-v2":"300","Hopper-v2":"300","Humanoid-v2":"500","Pusher-v2":"100"}
        import mbrl
        env = gym.make(env_name)
        if env_name == "Hopper-v2":
            ag = ars.ARSTrainer(
                    env=env_name,
                )
        else:
            ag = ars.ARSTrainer(
                    config={
                        "framework": "torch",
                    },
                    env=env_name,
                )
        tmp_n = checkpoint_num[env_name][1:] if checkpoint_num[env_name][0]=="0" else checkpoint_num[env_name]
        ag.restore(path+"checkpoint_000"+checkpoint_num[env_name]+"/"+"checkpoint-"+tmp_n)
        agent = lambda obs: ag.compute_single_action(obs)
    elif algo == "sac":
        from spinup.utils.test_policy import load_policy_and_env, run_policy
        device = "cuda" if gpu else "cpu"
        _,agent = load_policy_and_env(path,device=device)
        env = gym.make(env_name)
    elif algo == "ppo":
        from spinup.utils.test_policy import load_policy_and_env, run_policy
        device = "cuda" if gpu else "cpu"
        _,agent = load_policy_and_env(path,device=device)
        env = gym.make(env_name)
    else:
        print("Algo not known", algo)
        raise("Algo not known")
    return agent,env



def play(env, trainer, times, algorithm, repeat = 16, level = 0):
    initial_reward = 0
    percent = 0
    total_rewards = []
    total_ep = 0
    rewards = []
    iter_ep = 10
    for repeat in range(repeat):
        print('action repeated:', repeat)
        total_rewards = []
        if algorithm == 'rtrl':
            prev_action = [np.zeros(env.action_space.shape[0])]*5
        for k in range(iter_ep):
            obs = env.reset()
            total_reward = 0
            total_ep += 1
            prev_action = np.zeros(env.action_space.shape[0])
            prev_action_list = [np.zeros(env.action_space.shape[0])]*5
            done = True
            for i in range(times):
                if algorithm == "rtrl":
                    action = trainer((obs,prev_action_list[i]))
                elif algorithm == "planet":
                    action = trainer(obs,prev_action,done)
                else:
                    action = trainer(obs)
                if repeat == 0:
                    obs, reward, done, info = env.step(action)
                    prev_action_list.append(action)
                else:
                    obs, reward, done, info = env.step(prev_action)
                    prev_action_list.append(prev_action)
                total_reward += reward
                if repeat:
                    for j in range(repeat-1):
                        obs, reward, done, info = env.step(prev_action)
                        prev_action_list.append(prev_action)
                        total_reward += reward
                        if done:
                            total_rewards.append(total_reward)  
                            # print(total_reward)
                            break 
                else:        
                    if done:
                        total_rewards.append(total_reward)
                        # print(total_reward)
                        obs = env.reset()
                        prev_action = np.zeros(env.action_space.shape[0])
                        break
                if done:
                    prev_action = np.zeros(env.action_space.shape[0])
                    total_rewards.append(total_reward)  
                    break 
                prev_action = action
                
            env.close()
            
        rewards.append(reward_ave)
        env.close()


    return rewards

  
if __name__ == "__main__":

    # Input arguments from command line.
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument("--path", required=True, help="Filepath to trained checkpoint",
                        default="/app/data/ray_results/2/ARS_CartPole-v0_661d3_00000_0_2022-03-31_10-07-40/checkpoint_000100/checkpoint-100")
    parser.add_argument("--algo", required=True, help="Algorithm used", default="ars")
    parser.add_argument("--evaseed", required=True, help="Evaluation seed.",
                        default=1)
    parser.add_argument("--gpu", required=True, help="Evaluation seed.",
                        default=0)
    args = vars(parser.parse_args())
    for i in ["HalfCheetah-v2","Hopper-v2","continuous_CartPole-v0","Humanoid-v2","Pusher-v2","dmc_walker_walk","dmc_cartpole_balance","dmc_cheetah_run"]:
        if i in args["path"]:
            env_name = i

    print("Input of argparse:", args)
             
    times = 100000
    repeat = 5

        
    agent,env = load(args["path"],args["algo"],env_name,gpu=int(args["gpu"]))
    env.seed(int(args["evaseed"]))
    rewards = play(env, agent, times, algorithm = args["algo"],repeat = repeat)

