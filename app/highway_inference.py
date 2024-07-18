import configparser
import random
import warnings

import gymnasium as gym
import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.video_utils import record_videos
from app.utilities.weights_handler import WeightsHandler

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
        "absolute": True,
        "order": "sorted",
        "vehicles_count": 4,  #max number of observable vehicles
        "normalize": False
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 18,  #max number of existing vehicles
    "duration": 60,  # [s]
    "initial_spacing": 2,
    "collision_reward": -10,  # The reward received when colliding with a vehicle.
    'normalize_reward': False,
    "reward_speed_range": [28, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0,
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

features = ["x", "y", "vx", "vy"]
maxSize_proportion = int(config_parser['tilings']['maxSize_proportion'])
maxSize = maxSize_proportion * config["observation"]["vehicles_count"] * len(features)
iht = IHT(maxSize)
numTilings = maxSize // maxSize_proportion

space_action_len = len(env.action_type.actions_indexes)
weights_handler = WeightsHandler(maxSize, space_action_len)

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

for i in range(10):
    state, info = env.reset(seed=(42 + i))
    action = env.action_type.actions_indexes["IDLE"]

    done = False
    truncated = False
    avg_speed = 0
    num_steps = 0
    env.configure(config)
    if False and i == 0:
        env = record_videos(env)
    while not done and not truncated:
        tiles_list = tiles(iht, numTilings, state.flatten().tolist())

        best_action = 0
        best_estimate = estimate(tiles_list, 0, weights)
        for a in range(1, space_action_len):
            actual_estimate = estimate(tiles_list, a, weights)
            if actual_estimate > best_estimate:
                best_estimate = actual_estimate
                best_action = a
        action = best_action

        num_steps += 1
        avg_speed = ((num_steps - 1) * avg_speed + state[0][2]) / num_steps

        state, reward, done, truncated, info = env.step(action)
    # -----------------------------END INFERENCE-------------------------------

    print(f"Inference {i} -> avg_speed: {avg_speed}, num_steps: {num_steps}")

    # ------------------------------- RESULTS -----------------------------
    # Algorithm
    # chosen: Episodic
    # semi - gradient
    # Sarsa
    # Inference
    # 0 -> avg_speed: 27.35845184326172, num_steps: 8
    # Inference
    # 1 -> avg_speed: 25.88056049346924, num_steps: 10
    # Inference
    # 2 -> avg_speed: 26.77548484802246, num_steps: 5
    # Inference
    # 3 -> avg_speed: 28.609594106674194, num_steps: 8
    # Inference
    # 4 -> avg_speed: 27.99984868367513, num_steps: 3
    # Inference
    # 5 -> avg_speed: 26.69931616101946, num_steps: 14
    # Inference
    # 6 -> avg_speed: 26.001998299046566, num_steps: 19
    # Inference
    # 7 -> avg_speed: 24.981263478597004, num_steps: 3
    # Inference
    # 8 -> avg_speed: 23.399339294433595, num_steps: 10
    # Inference
    # 9 -> avg_speed: 27.072779655456543, num_steps: 2

    # Algorithm
    # chosen: Episodic
    # semi - gradient
    # Sarsa
    # Inference
    # 0 -> avg_speed: 27.35845184326172, num_steps: 8
    # Inference
    # 1 -> avg_speed: 25.88056049346924, num_steps: 10
    # Inference
    # 2 -> avg_speed: 26.77548484802246, num_steps: 5
    # Inference
    # 3 -> avg_speed: 28.609594106674194, num_steps: 8
    # Inference
    # 4 -> avg_speed: 28.602536916732788, num_steps: 8
    # Inference
    # 5 -> avg_speed: 26.282188415527344, num_steps: 7
    # Inference
    # 6 -> avg_speed: 24.397873458862303, num_steps: 25
    # Inference
    # 7 -> avg_speed: 24.89984718958537, num_steps: 12
    # Inference
    # 8 -> avg_speed: 24.063657760620117, num_steps: 11
    # Inference
    # 9 -> avg_speed: 23.044496096097507, num_steps: 39

