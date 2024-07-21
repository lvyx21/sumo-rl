import argparse
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import ray
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import sumo_rl

# 设置仿真环境变量
os.environ["SDL_VIDEODRIVER"] = "dummy"

# 解析命令行参数
parser = argparse.ArgumentParser(description="Render pretrained policy loaded from checkpoint")
parser.add_argument("checkpoint_dir", help="Directory containing checkpoint files.")
args = parser.parse_args()

checkpoint_dir = os.path.expanduser(args.checkpoint_dir)

# 注册SUMO环境
env_name = "two_intersections"
register_env(
    env_name,
    lambda _: ParallelPettingZooEnv(
        sumo_rl.parallel_env(
            net_file="sumo_rl/nets/two_intersections/two_intersections.net.xml",
            route_file="sumo_rl/nets/two_intersections/two_intersections.flow.rou.xml",
            out_csv_name="outputs/two_intersections/ppo",
            use_gui=False,
            num_seconds=80000,
        )
    ),
)

# 初始化Ray
ray.init()

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

PPOagent = PPO(config=config)

# 从checkpoint恢复PPO模型
PPOagent.restore(checkpoint_dir)

# 运行仿真并记录运行数据
env = ParallelPettingZooEnv(
    sumo_rl.parallel_env(
        net_file="sumo_rl/nets/two_intersections/two_intersections.net.xml",
        route_file="sumo_rl/nets/two_intersections/two_intersections.flow.rou.xml",
        out_csv_name="outputs/two_intersections/ppo",
        use_gui=True,
        num_seconds=80000,
    )
)

reward_sum = 0
frame_list = []
observations, infos = env.reset()

while True:
    actions = {}
    for agent in env.par_env.agents:
        action = PPOagent.compute_single_action(observations[agent])
        actions[agent] = action

    observations, rewards, terminations, truncations, infos = env.step(actions)
    reward_sum += sum(rewards.values())

    if env.render_mode == "rgb_array":
        img = Image.fromarray(env.render())
        frame_list.append(img)

    if all(terminations.values()) and all(truncations.values()):
        break

env.close()

print(f"Total Reward: {reward_sum}")
if frame_list:
    frame_list[0].save("out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)
