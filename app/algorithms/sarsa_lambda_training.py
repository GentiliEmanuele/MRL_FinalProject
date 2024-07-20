
import random

import gymnasium as gym
import numpy as np
import warnings

import app.utilities.config_utils as cu
import app.utilities.serialization_utils as su

from app.tile_coding.my_tiles import IHT, tiles
from app.utilities.video_utils import record_videos
from app.utilities.weights_handler import WeightsHandler

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")

config = cu.get_current_config()

env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)
state, info = env.reset(seed=cu.get_seed())
np.random.seed(cu.get_seed())
random.seed(cu.get_seed())

done = False
truncated = False

maxSize = cu.get_max_size()
iht = IHT(maxSize)
space_action_len = len(env.action_type.actions_indexes)
numTilings = cu.get_num_tilings()
alpha = cu.get_alpha() # step size
epsilon_0 = cu.get_epsilon0()
epsilon = epsilon_0
gamma = cu.get_gamma()
lambda_ = cu.get_lambda()
num_Episodes = 1000

weights_handler = WeightsHandler(maxSize, space_action_len)
weights = weights_handler.generate_weights()



print(iht.size)

for episode in range(num_Episodes):
    done = False
    truncated = False

    # initialization S
    state, info = env.reset(seed=cu.get_seed()+episode)

    # initialization x
    tiles_list = tiles(iht, numTilings, state.flatten().tolist())

    # initialization A
    action = cu.get_GLIE_action(epsilon, tiles_list, weights, random, episode, num_Episodes, env)

    # initialization z
    traces = np.zeros((maxSize, space_action_len))

    if episode == num_Episodes - 20:
        env = record_videos(env)
    num_steps = 0

    while not done and not truncated:
        # Take action, observe next stat and reward
        state_p, reward, done, truncated, info = env.step(action)

        # initialization x'
        tiles_list_p = tiles(iht, numTilings, state_p.flatten().tolist())

        # delta
        delta = reward

        # loop on tiles
        traces = np.zeros((maxSize, space_action_len))
        for tile in tiles_list:
            delta = delta - weights[tile, action]
            # traces[tile, action] += 1
            traces[tile, action] = 1

        if done or truncated:
            weights += alpha * delta * traces

        else:
            # choose A'
            # action_p = get_e_greedy_action(epsilon, space_action_len, tiles_list_p, weights, random)
            action_p = cu.get_GLIE_action(epsilon, tiles_list_p, weights, random, episode, num_Episodes, env)

            # loop on new tiles
            for tile_p in tiles_list_p:
                delta += gamma * weights[tile_p, action_p]

            weights += alpha * delta * traces
            traces = gamma * lambda_ * traces
            state = state_p
            action = action_p
            num_steps += 1

    print(f"Episode: {episode}, Num steps: {num_steps}")

print(f"IHT usage: {iht.count()}/{iht.size}")
weights_handler.save_weights(weights, "weights/sarsa_lambda_weights")
su.serilizeIHT(iht, "ihts/sarsa_lambda_iht.pkl")
env.close()

