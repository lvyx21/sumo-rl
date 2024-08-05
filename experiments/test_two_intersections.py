import argparse
import os
import sys
import csv

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
from pettingzoo.sisl import waterworld_v3
from ray.tune.registry import register_env
import sumo_rl
import traci
import csv

# 设置仿真环境变量

# 解析命令行参数
#parser = argparse.ArgumentParser(description="Render pretrained policy loaded from checkpoint")
#parser.add_argument("checkpoint_dir", help="Directory containing checkpoint files.")
#args = parser.parse_args()
if __name__ == "__main__":
    checkpoint_dir = "/home/lvyx/ray_results/PPO/PPO_two_intersections_f7cba_00000_0_2024-08-05_20-55-18/checkpoint_000017"
    ray.init()
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
            use_gui=False,
            num_seconds=80000, 
        )
    )

    reward_sum = 0
    obs, infos = env.reset()
    state=None
    done=False
    final_timeloss_data = {}
    #traci.start(["sumo", "-c", "sumo_rl/nets/two_intersections/two_intersections.sumocfg"])

    while True:
        action_dict = {}
        
        # 为每个代理计算动作和状态
        for agent_id in obs:
            action= PPOagent.compute_single_action(obs[agent_id], state)
            action_dict[agent_id] = action
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        reward_sum += sum(rewards.values())

        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            timeloss = traci.vehicle.getTimeLoss(vehicle_id)
            final_timeloss_data[vehicle_id]=timeloss 
        if all(terminations.values()) and all(truncations.values()):
            break
        

    env.close()
    with open("final_timeloss_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Vehicle ID", "Final TimeLoss"])
        for vehicle_id, final_timeloss in final_timeloss_data.items():
            writer.writerow([vehicle_id, final_timeloss])


    print(f"Total Reward: {reward_sum}")
