
import random

import gymnasium as gym
import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
import warnings
from matplotlib import pyplot as plt

from app.utilities.video_utils import record_videos

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")

features = ["x", "y", "vx"]

config = {
    "observation": {
        "type": "Kinematics",
        "features": features,
        "absolute": False,
        "order": "sorted",
        "vehicles_count": 4, #max number of observable vehicles
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 10, # max number of existing vehicles
    "duration": 72,  # [s]
    "initial_spacing": 2,
    "collision_reward": -50,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30], # [m/s] The reward for high speed is mapped linearly from this range to [0,
    # HighwayEnv.HIGH_SPEED_REWARD].
    "high_speed_reward": 0.3,
    "normalize_reward": True,
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

done = False
truncated = False

maxSize = 512 * 9
iht = IHT(maxSize)
space_action_len = len(env.action_type.actions_indexes)
weights = np.zeros(shape=(maxSize, space_action_len))
numTilings = maxSize // 128 # according to Sutton example we keep the ratio between maxSize and numTilings as 1 / 156
#alpha = 0.1 / numTilings # step size
alpha = 0.2
epsilon_0 = 0.1
epsilon = epsilon_0
gamma = 1 #0.9
lambda_ = 0.9
num_Episodes = 300

print(iht.size)

for episode in range(num_Episodes):
    print("Episode", episode)
    done = False
    truncated = False

    # initialization x, z, v_old
    tiles_list = tiles(iht, numTilings, state.flatten().tolist())
    traces = np.zeros(weights.shape)
    V_old = 0

    if episode == num_Episodes - 1:
        env = record_videos(env)

    if episode == num_Episodes - 20:
        env = record_videos(env)

    state, info = env.reset(seed=44+episode)
    num_steps = 0

    while not done and not truncated:
        # Choose A
        if random.random() < epsilon:
            action = random.randint(0, space_action_len - 1)
        else:
            best_action = 0
            best_estimate = estimate(tiles_list, 0, weights)
            for a in range(1, space_action_len):
                actual_estimate = estimate(tiles_list, a, weights)
                if actual_estimate > best_estimate:
                    best_estimate = actual_estimate
                    best_action = a
            action = best_action

        # Take action A, observe R, S'
        state_p, reward, done, truncated, info = env.step(action)

        # x'
        tiles_list_p = tiles(iht, numTilings, state_p.flatten().tolist())

        # v
        V = 0
        for tile in tiles_list:
            V = V + weights[tile, action]

        # v'
        V_p = 0
        for tile_p in tiles_list_p:
            V_p = V_p + weights[tile_p, action]

        # delta
        d = reward + gamma * V_p - V

        # traces
        traces = gamma * lambda_ * traces
        temp = 0
        for tile in tiles_list:
            temp += traces[tile, action]
        for tile in tiles_list:
            traces[tile, action] = traces[tile, action] + (1 - alpha * gamma * lambda_ * temp)

        # weights
        weights = weights + alpha * (d + V - V_old) * traces
        for tile in tiles_list:
            weights[tile, action] = weights[tile, action] - alpha * (V - V_old)

        # update old variables
        V_old = V_p
        tiles_list = tiles_list_p
        num_steps += 1

print(f"IHT usage: {iht.count()}/{iht.size}")
env.close()

