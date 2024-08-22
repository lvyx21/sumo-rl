"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register

from .ParallelPettingZooEnv import ParallelPettingZooEnv

register(
    id="sumo-rl-v0",
    entry_point="sumo_rl.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
register(
    id="sumo-rl-multi-v0",
    entry_point="sumo_rl.environment.env:SumoEnvironmentPZ",
    kwargs={"single_agent": False},
)

