import os
import random

import gymnasium as gym
import numpy as np
import warnings
import app.utilities.serialization_utils as su
from app.utilities.config_utils import ConfigUtils

from app.utilities.video_utils import record_videos
from app.utilities.weights_handler import WeightsHandler
from app.tile_coding.my_tiles import IHT, tiles, estimate

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")

env = gym.make('highway-v0', render_mode='rgb_array')

cu = ConfigUtils()
config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, lambda_, num_Episodes = cu.get_current_config()
print(f"Using configuration: {filename_suffix}")
iht = IHT(maxSize)
space_action_len = len(env.action_type.actions_indexes)

env.configure(config)
state, info = env.reset(seed=cu.get_seed())
np.random.seed(cu.get_seed())
random.seed(cu.get_seed())

done = False
truncated = False

weights_handler = WeightsHandler(maxSize, space_action_len)
weights = weights_handler.generate_weights()


for episode in range(num_Episodes):
    done = False
    truncated = False
    beta = 1 / (episode + 1)
    avg_reward = 0

    # initialization x, z, v_old
    state, info = env.reset(seed=cu.get_seed()+episode)
    tiles_list = tiles(iht, numTilings, state.flatten().tolist())
    traces = np.zeros((maxSize, space_action_len))
    V_old = 0

    if episode == num_Episodes - 10:
        env = record_videos(env)
    num_steps = 0

    while not done and not truncated:
        # Choose A
        action = cu.get_e_greedy_action(epsilon, tiles_list, weights, random, env)

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
        d = reward - avg_reward + gamma * V_p - V
        # update avg_reward
        avg_reward = avg_reward + beta * d

        # traces
        temp = 0
        for tile in tiles_list:
            temp += traces[tile, action]
        traces = gamma * lambda_ * traces
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

    print(f"Episode: {episode}, Num steps: {num_steps}")

print(f"IHT usage: {iht.count()}/{iht.size}")

os.makedirs("weights", exist_ok=True)
os.makedirs("ihts", exist_ok=True)
weights_handler.save_weights(weights, f"weights/true_online_td_lambda_differential_weights{filename_suffix}")
su.serilizeIHT(iht, f"ihts/true_online_td_lambda_differential_iht{filename_suffix}.pkl")
env.close()

