"""SUMO Environment for Traffic Signal Control."""
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
from .vehicle_controller import VehicleController
from gymnasium import spaces

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(gym.Env,VehicleController):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The duration in seconds of the simulation. Default: 20000
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict): String with the name of the reward function used by the agents, a reward function, or dictionary with reward functions assigned to individual traffic lights by their keys.
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict] = "custom-function",
        observation_class: ObservationFunction = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
        tripinfo_file: str = "tripinfo.xml"
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None
        self.sumo=None
        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        self.tripinfo_file = tripinfo_file 
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None
        
        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net,"--tripinfo-output",self.tripinfo_file])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net,"--tripinfo-output",self.tripinfo_file], label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        self.ts_ids = list(conn.trafficlight.getIDList())
        self.vehicle_ids=traci.vehicle.getIDList()
        self.observation_class = observation_class  
        self.smart_vehicle= {"default_vehicle": VehicleController(self, "default_vehicle", self.sumo)}
        self.known_smart_vehicle_id=[]
        self.online_vehicle_name="online_smart_vehicle"
        self.last_type2_vehicle = None
        self.vehicle_assignment_index = 0
        self.agent_ids=self.ts_ids+self.known_smart_vehicle_id
        #self.vehicle_observation_space=spaces.Box(low=0,high=1,shape=(4,),dtype=np.float32)
        #self.vehicle_action_space=spaces.Discrete(3)
        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    conn,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    conn,
                )
                for ts in self.ts_ids
            }
        #always smart vehicle
        
        # WZ: we need to initialize the smart vehicle agent at the very beginning. 
        # Adding controllable smart vehicle is the next step to be done in the update.
        # Therefore, we should not add if sentence as well as appending smart vehicle id here. These should be done later.
        for vehicle_id in self.vehicle_ids:
            if conn.vehicle.getTypeID(vehicle_id)=='type2': # WZ: when will the vehicle type2 be assigned? In the whole repo, I only found you check if vehicle type is type2.
                self.known_smart_vehicle_id.append(vehicle_id)
                self.smart_vehicle= {vehicle_id: VehicleController(
                    self,
                    vehicle_id,
                    conn,
                )}
        
                    
                
            
        
        

    
        

        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
        if self.smart_vehicle:
            for vehicle_id in self.known_smart_vehicle_id:
                self.rewards[vehicle_id]=None

        self.traffic_light_states={}

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
            "--tripinfo-output", 
            self.tripinfo_file,
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")
        
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        #self._assign_next_type2_vehicle()
        # if self.use_gui or self.render_mode is not None:
        #     if "DEFAULT_VIEW" not in dir(traci.gui):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
        #         traci.gui.DEFAULT_VIEW = "View #0"
        #     self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)
       
        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    self.sumo,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    self.sumo,
                )
                for ts in self.ts_ids
            }
        self.smart_vehicle={}
        for vehicle_id in self.vehicle_ids:
            if self.sumo.vehicle.getTypeID(vehicle_id)=='type2':
                self.known_smart_vehicle_id.append(vehicle_id)
                self.smart_vehicle [vehicle_id]= VehicleController(
                    self,
                    vehicle_id,
                    self.sumo,
                    )

        self.vehicles = dict()
        vehicle_ids = self.sumo.vehicle.getIDList()
        self.vehicles.update ({vehicle_id: {} for vehicle_id in vehicle_ids})
        
        
            
        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()



    def step(self, action: Union[dict, int]):
        # No action, follow fixed TL defined in self.phases
        
        if self.fixed_ts or action is None or action == {}:
            for _ in range(self.delta_time):
                self._update_traffic_light()
        else:
            # 拆分红绿灯和车辆的动作
            traffic_light_action = {k: v for k, v in action.items() if k in self.ts_ids}
            self._apply_actions(traffic_light_action)
            vehicle_actions={k: v for k, v in action.items() if k in self.known_smart_vehicle_id}
            for vehicle_id,vehicle_action in vehicle_actions.items():
                if vehicle_id in self.known_smart_vehicle_id:
                    if self.smart_vehicle[vehicle_id].vehicle_time_to_act():
                        self.smart_vehicle[vehicle_id].set_next_action(vehicle_action)
            self._run_steps()

        #if self.last_type2_vehicle is not None and self.sumo.vehicle.getRoadID(self.last_type2_vehicle) == "":  # 到达终点
        #self._assign_next_type2_vehicle()
        
     
        observations = self._compute_observations()
        rewards = self._compute_rewards()

        with open("rewards_log.txt", "a") as f:
            f.write(f"Step {self.sim_step}:\n")
            f.write(f"Traffic Signal Rewards: { {ts_id: rewards[ts_id] for ts_id in self.ts_ids} }\n")
            f.write(f"Vehicle Rewards: { {vehicle_id: rewards[vehicle_id] for vehicle_id in self.known_smart_vehicle_id} }\n")
            f.write("\n")

        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info
    def _assign_next_type2_vehicle(self):
        with open("type2_vehicle_assignment.log","a")as log_file:
            target_route=('p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17')
            log_file.write(f"Current known_smart_vehicle_id: {self.known_smart_vehicle_id}\n")

            for vehicle_id in self.known_smart_vehicle_id:
                    if self.sumo.vehicle.getRoadID(vehicle_id) != "":
                        log_file.write(f"Type2 vehicle {vehicle_id} is still on the road, not assigning a new type2 vehicle.\n")
                        return
            vehicle_ids = self.sumo.simulation.getDepartedIDList()
        
            for vehicle_id in vehicle_ids:
                route=self.sumo.vehicle.getRoute(vehicle_id)
                log_file.write(f"Vehicle {vehicle_id} route: {route}\n")

                if route==target_route:
                    self.sumo.vehicle.setType(vehicle_id,"type2")
                    self.sumo.vehicle.setColor(vehicle_id,(0,255,0,255))
                    self.known_smart_vehicle_id.append(vehicle_id)
                    self.smart_vehicle[vehicle_id]=VehicleController(self,vehicle_id,self.sumo)
                    #agent
                    log_file.write(f"matched vehicle{vehicle_id} has been set to type2\n")
                    break
                else:
                    log_file.write(f"Skipped vehicle {vehicle_id} as its route does not match the target edges.\n")
                




    '''
    def _update_known_smart_vehicles(self):
        """Update the list of known smart vehicles (type2) at each step."""
        current_vehicle_ids = self.sumo.vehicle.getIDList()
        for vehicle_id in current_vehicle_ids:
            if self.sumo.vehicle.getTypeID(vehicle_id) == 'type2' and vehicle_id not in self.known_smart_vehicle_id:
                self.known_smart_vehicle_id.append(vehicle_id)
                self.smart_vehicle[vehicle_id] = VehicleController(self.sumo, self, vehicle_id)
                self.last_type2_vehicle = vehicle_id
        print("Known Smart Vehicle IDs:", self.known_smart_vehicle_id)

    def _get_traffic_light_state(self, tls_id):
        # 获取当前红绿灯的状态
        state = traci.trafficlight.getRedYellowGreenState(tls_id)
        queue_length = traci.edge.getLastStepVehicleNumber(tls_id)
        avg_speed = traci.edge.getLastStepMeanSpeed(tls_id)
        # 返回一个包含必要信息的字典
        return {'state': state, 'queue_length': queue_length, 'avg_speed': avg_speed}

    def _update_traffic_light(self, tls_id):
        # 获取其他红绿灯的状态信息
        other_tls_state = {k: v for k, v in self.traffic_light_states.items() if k != tls_id}
        # 基于自身和其他红绿灯的状态信息更新当前红绿灯
        self._make_decision(tls_id, other_tls_state)
    
    def _make_decision(self, tls_id, other_tls_state):
        current_state = self.traffic_light_states[tls_id]
        if current_state['queue_length'] > 10:  # 队列长度超过10辆车
            traci.trafficlight.setRedYellowGreenState(tls_id, 'G')  # 设置为绿灯
        else:
            traci.trafficlight.setRedYellowGreenState(tls_id, 'r')  # 设置为红灯
    '''

    def _run_steps(self):
        time_to_act = False
        vehicle_time_to_act=False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True
            '''
            for vehicle_id,vehicle in self.smart_vehicle.items():
                vehicle.update()
                if vehicle.vehicle_time_to_act():
                    vehicle_time_to_act=True
            '''
            for vehicle_id, vehicle in list(self.smart_vehicle.items()):
                vehicle.update()
                if vehicle.vehicle_time_to_act():
                    vehicle_time_to_act = True
                    with open("vehicle.log","a")as vehicle_file:
                        vehicle_file.write(f"vehicle acting now.\n")



        
            
                
 

    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                 if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)
                    
    
    

   
        


    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info

    def _compute_observations(self):
        self.observations.update(
            {
            ts: self.traffic_signals[ts].compute_observation()
            for ts in self.ts_ids
            if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
    
        # 更新智能车辆的观察值
        if self.known_smart_vehicle_id:
            obs=self.smart_vehicle[self.known_smart_vehicle_id[0]].compute_observation()
            self.observations[self.online_vehicle_name]=obs
        else:
            self.observations[self.online_vehicle_name]=np.zeros(3)

        vehicle_dic={
            self.online_vehicle_name: self.observations[self.online_vehicle_name].copy()
                #if self.smart_vehicle.vehicle_time_to_act
        }
        # 返回观察值的副本
        print({
            
                **{ts: self.observations[ts].copy()
                for ts in self.observations.keys()
                if ts in self.traffic_signals and (self.traffic_signals[ts].time_to_act or self.fixed_ts)
                },
                **vehicle_dic
        })
        return {
            
                **{ts: self.observations[ts].copy()
                for ts in self.observations.keys()
                if ts in self.traffic_signals and (self.traffic_signals[ts].time_to_act or self.fixed_ts)
                },
                **vehicle_dic
        }

      
    def _compute_rewards(self):
        self.rewards.update(
            {
                ts: self.traffic_signals[ts].compute_reward()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts 
            }
        )
        if self.known_smart_vehicle_id:
            self.rewards["online_smart_vehicle"]=self.smart_vehicle[self.known_smart_vehicle_id[0]].compute_reward()
        else:
            self.rewards["online_smart_vehicle"]=0.0     
        return self.rewards

    @property
    def observation_space(self):
        if self.single_agent:
            if self.agent_id in self.ts_ids:
                return self.traffic_signals[self.ts_ids[0]].observation_space
            elif self.agent_id in self.online_vehicle_name:
                return self.smart_vehicle[self.known_smart_vehicle_id[0]].observation_space
        else:
            return {agent_id: self.observation_spaces(agent_id) for agent_id in self.agent_ids}

    @property
    def action_space(self):
        if self.single_agent:
            if self.agent_id in self.ts_ids:
                return self.traffic_signals[self.ts_ids[0]].action_space
            elif self.agent_id in self.known_smart_vehicle_id:
                return self.smart_vehicle[self.known_smart_vehicle_id[0]].action_space
        else:
            return {agent_id: self.action_spaces(agent_id) for agent_id in self.agent_ids}

    def observation_spaces(self, agent_id: str):
        if agent_id in self.ts_ids:
            return self.traffic_signals[agent_id].observation_space
        else:
            return self.smart_vehicle["default_vehicle"].observation_space # WZ: you will be restricted to only one vehicle agent by assigning obs space like this, although good for now.

    def action_spaces(self, agent_id: str) -> gym.spaces.Discrete:
        if agent_id in self.ts_ids:
            return self.traffic_signals[agent_id].action_space
        else:
            return self.smart_vehicle["default_vehicle"].action_space
        
    

    def _sumo_step(self):
        self.sumo.simulationStep()

    

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        """Encode the state of the traffic signal into a hashable object."""
        phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1 :]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)


class SumoEnvironmentPZ(AECEnv, EzPickle):
    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.

    For more information, see https://pettingzoo.farama.org/api/aec/.

    The arguments are the same as for :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs
        self.seed()
        self.env = SumoEnvironment(**self._kwargs)
        self.render_mode = self.env.render_mode
        #self.online_vehicle_name="online_smart_vehicle"
        self.agents = self.env.ts_ids+[self.env.online_vehicle_name]
        self.possible_agents = self.env.ts_ids+[self.env.online_vehicle_name]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # self.default_vehicle="default_vehicle"
        #self.smart_vehicle= {"default_vehicle": VehicleController(self.env, "default_vehicle", self.env.sumo)}
        
        # spaces
        #max_actions = max([self.env.action_spaces(agent).n for agent in self.agents])
        #self.action_spaces[agent] = Discrete(max_discrete_action_space)

        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        """Set the seed for the environment."""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self.env.reset(seed=seed, options=options)
        #current_known_smart_vehicle_ids = [str(vehicle_id) for vehicle_id in self.env.known_smart_vehicle_id]
        #self.agents = self.env.ts_ids + self.env.known_smart_vehicle_id
        #if self.env.known_smart_vehicle_id:
            #self.agents=self.env.ts_ids+[self.online_smart_vehicle]
            #self.online_vehicle_id=self.env.known_smart_vehicle_id[0]
        #else:
            #self.agents=self.env.ts_ids
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()
        #self.smart_vehicle = {}
        #for vehicle_id in self.env.known_smart_vehicle_id:
            #self.smart_vehicle[vehicle_id] = VehicleController(self.env, vehicle_id, self.env.sumo)

    def update_online_vehicle(self):
        """Update the ID that the fixed vehicle name refers to."""
        if self.env.known_smart_vehicle_id:
            self.online_vehicle_id = self.env.known_smart_vehicle_id[0]
        else:
            self.online_vehicle_id = None
    def compute_info(self):
        """Compute the info for the current step."""
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        """Return the observation space for the agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for the agent."""
        return self.action_spaces[agent]

    def observe(self, agent):
        """Return the observation for the agent."""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.env.close()

    def render(self):
        """Render the environment."""
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file."""
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        self.env._assign_next_type2_vehicle()
        self.update_online_vehicle()
        #current_known_smart_vehicle_ids = [str(vehicle_id) for vehicle_id in self.env.known_smart_vehicle_id]
        with open("tagent.log","a")as agent_file:
            agent_file.write(f"agent_id: {self.agents}\n")
        '''
        for new_agent in self.env.known_smart_vehicle_id:
            self.rewards[new_agent] = 0
            self.terminations[new_agent] = False
            self.truncations[new_agent] = False
            self.infos[new_agent] = {}
            self.action_spaces[new_agent] = self.env.action_spaces(new_agent)
            self.observation_spaces[new_agent] = self.env.observation_spaces(new_agent)
            #self._cumulative_rewards[new_agent] = 0
        '''    
        """Step the environment."""
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        with open("tagent.log","a")as agent_file:
            agent_file.write(f"agent: {agent}\n")
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(agent, self.action_spaces[agent].n, action)
            )

        if not self.env.fixed_ts:
            if agent in self.env.ts_ids:
                self.env._apply_actions({agent: action})
            elif agent == self.env.online_vehicle_name:
                if self.env.known_smart_vehicle_id:
                    self.env.smart_vehicle[self.env.known_smart_vehicle_id[0]].set_next_action(action)

        if self._agent_selector.is_last():
            if not self.env.fixed_ts:
                self.env._run_steps()
            else:
                for _ in range(self.env.delta_time):
                    self.env._sumo_step()

            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        with open("tagent.log","a")as agent_file:
            agent_file.write(f"agentnext: {self.agent_selection}\n")
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

