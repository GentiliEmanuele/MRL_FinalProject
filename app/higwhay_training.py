import random

import gymnasium as gym
import numpy as np

import warnings
import configparser

from app.algorithms.episodic_semi_gradient_sarsa import episodic_semi_gradient_sarsa
from app.algorithms.true_online_td_lambda import true_online_td_lambda

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")


# Create a ConfigParser object
config_parser = configparser.ConfigParser()
# Read the configuration file
config_parser.read('config.ini')

config = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "absolute": False,
        "order": "sorted",
        "vehicles_count": 4,  #max number of observable vehicles
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 5,  #max number of existing vehicles
    "duration": 36,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0,
    # HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 1200,  # [px]
    "screen_height": 250,  # [px]
    "centering_position": [0.1, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": False,
    "offscreen_rendering": False
}

env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)
state, info = env.reset(seed=44)
np.random.seed(44)
random.seed(44)

algorithm_type = int(config_parser['algorithm']['type'])
if algorithm_type == 1:
    print("Algorithm chosen: Episodic semi-gradient Sarsa")
    algorithm_instance = episodic_semi_gradient_sarsa()
elif algorithm_type == 2:
    print("Algorithm chosen: True online TD(lambda)")
    algorithm_instance = true_online_td_lambda()
else:
    print("Invalid configuration.\nAlgorithm chosen: Episodic semi-gradient Sarsa")
    algorithm_instance = episodic_semi_gradient_sarsa()

weights = algorithm_instance.execute(env=env, config=config)

env.close()
