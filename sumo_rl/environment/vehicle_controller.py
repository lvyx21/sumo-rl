import os
import sys
from typing import Callable, List, Union

from gymnasium.spaces import Box

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal



LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces
class VehicleController:
    MIN_GAP = 2.5

    def __init__(self, sumo, env,vehicle_id):
        self.vehicle_id=vehicle_id
        self.sumo = sumo
        self.env = env
        self.id = vehicle_id
        self.current_speed = self.sumo.vehicle.getSpeed(self.id)
        self.next_action_time = env.sim_step
        self.delta_time=5
        self.num_surrounding_vehicles=5
        #num_traffic_signals=len(self.env.ts_ids)
        #num_observations=4+len(self.env.ts_ids)+self.num_surrounding_vehicles*4+len(self.env.ts_ids)*2
        num_observations=5+len(self.env.ts_ids)
        self.observation_space=spaces.Box(low=0,high=1,shape=(num_observations,),dtype=np.float32)
        #self.action_space = Box(low=-self.sumo.vehicle.getDecel(self.id), high=self.sumo.vehicle.getAccel(self.id), shape=(1,), dtype=np.float32)

        self.action_space = Box(low=-4, high=4, shape=(1,), dtype=np.float32)
        self.next_action=None
        

    def update(self):
        self.current_speed=self.sumo.vehicle.getSpeed(self.id)
        if self.vehicle_time_to_act():
            self.excute_action() 
    
    def execute_action(self):
        if self.next_action is not None:
            self.set_next_action(self.next_action)
            self.next_action=None

    def set_next_action(self, action):
        """
        Sets the next action for the vehicle which could be accelerate, maintain, or decelerate.

        Args:
            action (str): One of ['accelerate', 'maintain', 'decelerate']
        """
        if self.env.sim_step < self.next_action_time:
            return  # 未到下一次行为设置时间
        new_speed = self.current_speed + action * self.delta_time
        self.sumo.vehicle.setSpeed(self.id, new_speed)
        self.next_action_time = self.env.sim_step + self.delta_time
        #setspeed可能会导致车辆光速穿过

    def vehicle_time_to_act(self):
        return self.next_action_time <= self.env.sim_step

    
    
    def get_observation(self):
        max_speed = self.sumo.vehicle.getMaxSpeed(self.id)
        max_length = self.sumo.lane.getLength(self.sumo.vehicle.getLaneID(self.id))
        
        # 获取当前车辆的速度
        this_speed = self.sumo.vehicle.getSpeed(self.id)
        
        # 获取领先车辆的信息
        lead_id = self.sumo.vehicle.getLeader(self.id)
        if lead_id in ["", None]:
            lead_speed = max_speed
            lead_head = max_length
        else:
            lead_speed = self.sumo.vehicle.getSpeed(lead_id)
            lead_head = self.sumo.vehicle.getPosition(lead_id)[0] - self.sumo.vehicle.getPosition(self.id)[0] - self.sumo.vehicle.getLength(self.id)
        
        # 获取跟随车辆的信息
        follower_id = self.sumo.vehicle.getFollower(self.id)
        if follower_id in ["", None]:
            follow_speed = 0
            follow_head = max_length
        else:
            follow_speed = self.sumo.vehicle.getSpeed(follower_id)
            follow_head = self.sumo.vehicle.getHeadway(follower_id)
        
        # 归一化并组合观测
        observation = [
            this_speed / max_speed,
            (lead_speed - this_speed) / max_speed,
            lead_head / max_length,
            (this_speed - follow_speed) / max_speed,
            follow_head / max_length
        ]

        for ts_id in self.env.ts_ids:
            traffic_signal_state = self.sumo.trafficlight.getRedYellowGreenState(ts_id)
            observation.append(traffic_signal_state)

        return np.array(observation, dtype=np.float32)
        
    
        
    def compute_reward(self):
        """
        Compute the reward for the vehicle.
        """
        '''
        # 示例奖励函数：根据车辆速度给予奖励，速度越高奖励越高
        speed = self.sumo.vehicle.getSpeed(self.id)
        waiting_time=self.sumo.vehicle.getWaitingTime(self.id)
        time_loss = self.sumo.vehicle.getTimeLoss(self.id) 
        type2_waiting_time = sum(self.sumo.vehicle.getWaitingTime(veh) for veh in self.sumo.vehicle.getIDList() if self.sumo.vehicle.getTypeID(veh) == 'type2')
        type2_time_loss = sum(self.sumo.vehicle.getTimeLoss(veh) for veh in self.sumo.vehicle.getIDList() if self.sumo.vehicle.getTypeID(veh) == 'type2')
        collision_reward = 0
        collision_weight = 1000 
        collisions = self.sumo.simulation.getCollidingVehiclesIDList()
        if self.id in collisions:
            collision_reward = -1 * collision_weight
    
        reward =-time_loss+speed/self.sumo.vehicle.getMaxSpeed(self.id)-waiting_time+collision_reward
        return reward
        '''
            
        """
        Compute the reward for the vehicle with type id = 'type2'.
        """
        # 检查车辆类型
        if self.sumo.vehicle.getTypeID(self.id) != 'type2':
            return 0  # 非 type2 类型的车辆不计算奖励

        # 获取当前车辆的速度和加速度
        speed = self.sumo.vehicle.getSpeed(self.id)
        max_speed = self.sumo.vehicle.getMaxSpeed(self.id)
        acceleration = self.sumo.vehicle.getAcceleration(self.id)
        max_acceleration = 4.0  # 假设最大加速度为4 m/s²

        # 获取同一路上所有 type2 车辆的速度和加速度
        vehicles_on_same_road = [
            veh for veh in self.sumo.vehicle.getIDList()
            if self.sumo.vehicle.getRoadID(veh) == self.sumo.vehicle.getRoadID(self.id) and self.sumo.vehicle.getTypeID(veh) == 'type2'
        ]
        speeds = [self.sumo.vehicle.getSpeed(veh) for veh in vehicles_on_same_road]
        accelerations = [self.sumo.vehicle.getAcceleration(veh) for veh in vehicles_on_same_road]

        num_vehicles = len(vehicles_on_same_road)
        total_waiting_time = sum(self.sumo.vehicle.getWaitingTime(veh) for veh in vehicles_on_same_road)

        # 计算 r1：速度偏差惩罚
        speed_deviation_sum = sum(max_speed - v for v in speeds if v <= max_speed)
        r1 = -speed_deviation_sum / (max_speed * num_vehicles) if num_vehicles > 0 else 0

        # 计算 r2：加速度惩罚
        accel_penalty_sum = sum((a / max_acceleration) ** 2 if a < 0 else a for a in accelerations)
        r2 = -accel_penalty_sum / num_vehicles if num_vehicles > 0 else 0

        # 计算 r3：等待时间惩罚
        r3 = -total_waiting_time / num_vehicles if num_vehicles > 0 else 0

        # 计算碰撞惩罚
        collision_reward = 0
        collision_weight = 1000
        collisions = self.sumo.simulation.getCollidingVehiclesIDList()
        if self.id in collisions:
            collision_reward = -1 * collision_weight

        stops = sum(1 for veh in self.sumo.vehicle.getIDList() if self.sumo.vehicle.getSpeed(veh) < 0.1)

        # 总奖励
        reward = r1 + r2 + r3 + collision_reward-stops
        return reward


    
