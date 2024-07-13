import configparser
import random

import gymnasium as gym
import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.video_utils import record_videos
from app.utilities.weights_handler import WeightsHandler


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
    "duration": 160,  # [s]
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

maxSize = 256 * 12
space_action_len = len(env.action_type.actions_indexes)
weights_handler = WeightsHandler(maxSize, space_action_len)
iht = IHT(maxSize)
numTilings = maxSize // 256

algorithm_type = int(config_parser['algorithm']['type'])
if algorithm_type == 1:
    print("Algorithm chosen: Episodic semi-gradient Sarsa")
    filename = "algorithms/weights/episodic_semi_gradient_sarsa_weights.npy"
elif algorithm_type == 2:
    print("Algorithm chosen: True online TD(lambda)")
    filename = "algorithms/weights/true_online_td_lambda_weights.npy"
else:
    print("Invalid configuration.\nAlgorithm chosen: Episodic semi-gradient Sarsa")
    filename = "algorithms/weights/episodic_semi_gradient_sarsa_weights.npy"

weights = weights_handler.load_weights(filename)

if weights is None:
    print('Error in weights loading')
    exit(0)

done = False
truncated = False
state, info = env.reset(seed=44)
env.configure(config)
env = record_videos(env)

while not done and not truncated:
    tiles_list = tiles(iht, numTilings, state.flatten().tolist())

    best_action = 0
    best_estimate = 0
    for a in range(0, space_action_len):
        actual_estimate = estimate(tiles_list, a, weights)
        if actual_estimate > best_estimate:
            best_estimate = actual_estimate
            best_action = a
    action = best_action

    state, reward, done, truncated, info = env.step(action)