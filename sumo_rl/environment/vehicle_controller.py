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

    def __init__(self, env,vehicle_id,sumo,):
        self.vehicle_id=vehicle_id
        self.env = env
        self.id = vehicle_id
        self.sumo = sumo
        self.current_speed = 0 #self.sumo.vehicle.getSpeed(self.id)
        self.next_action_time = 0 #env.sim_step
        self.delta_time=5
        self.num_surrounding_vehicles=5
        num_observations=5+len(self.env.ts_ids)
        self.observation_fn = self.env.observation_class(self)
        #自定义
        #self.observation_space = self.observation_fn.observation_space()
        self.observation_space=spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)
        self.next_action=None
        

    def update(self):
        self.current_speed=self.sumo.vehicle.getSpeed(self.id)
        if self.vehicle_time_to_act():
            self.execute_action() 
    
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

    def one_hot_encode_traffic_signal(self,state_str):
        """将交通信号字符串转换为 one-hot 编码."""
        mapping = {'r': [1, 0, 0, 0],  # 红灯
                'y': [0, 1, 0, 0],  # 黄灯
                'g': [0, 0, 1, 0],  # 绿灯（小写表示一般绿灯）
                'G': [0, 0, 0, 1],
                's': [0, 0, 0, 0]}  # 绿灯（大写表示特殊绿灯或直行绿灯）
        
        # 对字符串中的每个字符进行 one-hot 编码
        one_hot_encoded = [mapping[char] for char in state_str]
        
        # 将所有 one-hot 向量展平（flatten）为一个长向量
        return np.array(one_hot_encoded).flatten()
    
    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()
    def _observation_fn_default(self):
        max_speed = self.sumo.vehicle.getMaxSpeed(self.id)
        max_length = self.sumo.lane.getLength(self.sumo.vehicle.getLaneID(self.id))
        
        # 获取当前车辆的速度
        this_speed = self.sumo.vehicle.getSpeed(self.id)
        speed_observe=this_speed/max_speed
        # 获取领先车辆的信息
        current_vehicles = self.sumo.vehicle.getIDList()
        lead_id = self.sumo.vehicle.getLeader(self.id)    
        if lead_id in current_vehicles:
            lead_speed = self.sumo.vehicle.getSpeed(lead_id)
            lead_head = self.sumo.vehicle.getPosition(lead_id)[0] - self.sumo.vehicle.getPosition(self.id)[0] - self.sumo.vehicle.getLength(self.id)
        else:
            lead_speed = max_speed
            lead_head = max_length
        lead_speed_observe=(lead_speed-this_speed)/max_speed
        lead_distance_observe=lead_head/max_length
        '''
        # 获取跟随车辆的信息
        follower_id = self.sumo.vehicle.getFollower(self.id)
        if follower_id in ["", None]:
            follow_speed = 0
            follow_head = max_length
        else:
            follow_speed = self.sumo.vehicle.getSpeed(follower_id)
            follow_head = self.sumo.vehicle.getHeadway(follower_id)
        '''
  
    
        #for ts_id in self.env.ts_ids:
            #traffic_signal_state = self.sumo.trafficlight.getRedYellowGreenState(ts_id)
            # 对每个交通信号状态字符串进行 one-hot 编码
            #encoded_state = self.one_hot_encode_traffic_signal(traffic_signal_state)
           

        #observation=np.array(speed_observe+lead_speed_observe+lead_distance_observe+encoded_state, dtype=np.float32)
        observation=np.array(speed_observe+lead_speed_observe+lead_distance_observe, dtype=np.float32)
        return observation
    
        
    def compute_reward(self):

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


    
