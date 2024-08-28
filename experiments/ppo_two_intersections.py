import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
#from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from sumo_rl.environment.ParallelPettingZooEnv import ParallelPettingZooEnv
from ray.tune.registry import register_env
import sumo_rl
from supersuit import pad_observations_v0 
#from supersuit import pad_action_space_v0
 # 引入 SuperSuit 包装器


def create_env(env_config):
    env = sumo_rl.parallel_env(
        net_file="sumo_rl/nets/plymouth/plymouth_nixon_and_huron.net.xml",
        route_file="sumo_rl/nets/plymouth/plymouth_test.rou.xml",
        out_csv_name="outputs/two_intersections/ppo",
        use_gui= False,
        num_seconds=80000,
    )
    env = pad_observations_v0(env)
    #env=pad_action_space_v0(env)  # 使用 pad_observations_v0 包装器
    return env

if __name__ == "__main__":
    ray.init()

    env_name = "two_intersections"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(create_env({}))
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=5e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=5,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
