import random

import gymnasium as gym
import numpy as np
import warnings
import app.utilities.config_utils as cu
import app.utilities.serialization_utils as su

from app.utilities.video_utils import record_videos
from app.utilities.weights_handler import WeightsHandler
from app.tile_coding.my_tiles import IHT, tiles, estimate

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")

env = gym.make('highway-v0', render_mode='rgb_array')

cu = cu.ConfigUtils()
config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, lambda_, num_Episodes = cu.get_current_config()
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

    # initialization x, z, v_old
    state, info = env.reset(seed=cu.get_seed()+episode)
    tiles_list = tiles(iht, numTilings, state.flatten().tolist())
    traces = np.zeros((maxSize, space_action_len))
    V_old = 0

    if episode == num_Episodes - 20:
        env = record_videos(env)
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

        if done:
            reward = -50

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
weights_handler.save_weights(weights, f"weights/true_online_td_lambda_weights{filename_suffix}")
su.serilizeIHT(iht, f"ihts/true_online_td_lambda_iht{filename_suffix}.pkl")
env.close()