import random
import time

import gymnasium as gym
import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
from video_utils import record_videos
import warnings
from tabulate import tabulate
from matplotlib import pyplot as plt

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")

config = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "absolute": False,
        "order": "sorted",
        "vehicles_count": 4, #max number of observable vehicles
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 50, #max number of existing vehicles
    "duration": 36,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30], # [m/s] The reward for high speed is mapped linearly from this range to [0,
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

# See initial configuration
plt.imshow(env.render())
plt.show()
#env = record_videos(env)


features = ["x", "y", "vx", "vy"]

done = False
truncated = False

maxSize = 256 * 12
iht = IHT(maxSize)
space_action_len = len(env.action_type.actions_indexes)
weights = np.random.rand(maxSize, space_action_len)
numTilings = maxSize // 256 # according to Sutton example we keep the ratio between maxSize and numTilings as 1 / 156
alpha = 0.1 / numTilings # step size
epsilon_0 = 0.1
epsilon = epsilon_0
gamma = 0.9
num_Episodes = 100

# Choose A
action = env.action_type.actions_indexes["IDLE"]
for episode in range(num_Episodes):
    print("Episode", episode)
    done = False
    truncated = False
    state, info = env.reset(seed=44)
    if episode == num_Episodes - 1:
        config["duration"] = 160
        env = record_videos(env)
    while not done and not truncated:
        # tiles_list of initial state
        tiles_list = tiles(iht, numTilings, state.flatten().tolist())
        # Take action A, observe R, S'
        state_p, reward, done, truncated, info = env.step(action)
        if done:
            for tile in tiles_list:
                weights[tile, action] = weights[tile, action] + alpha * (reward - estimate(tiles_list, action, weights))
        else:
            tiles_list_p = tiles(iht, numTilings, state_p.flatten().tolist())
            # Choose A' as a function of q(s, ., w) (e.g e-greedy)
            if random.random() < epsilon:
                action_p = random.randint(0, space_action_len - 1)
            else:
                best_action = 0
                best_estimate = 0
                for a in range(0, space_action_len):
                    actual_estimate = estimate(tiles_list_p, a, weights)
                    if actual_estimate > best_estimate:
                        best_estimate = actual_estimate
                        best_action = a
                action_p = best_action
            for tile in tiles_list:
                weights[tile, action] = weights[tile, action] + alpha*(reward + gamma * estimate(tiles_list_p, action_p, weights) - estimate(tiles_list, action, weights))
            state = state_p
            action = action_p
        epsilon = epsilon - epsilon_0 / num_Episodes

print(weights)
env.close()
