import os
import random
import datetime

import gymnasium as gym
import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.config_utils import ConfigUtils
from app.utilities.video_utils import record_videos
import warnings

import app.utilities.serialization_utils as su

from app.utilities.weights_handler import WeightsHandler

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")

env = gym.make('highway-v0', render_mode='rgb_array')

cu = ConfigUtils()
config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, _, num_Episodes = cu.get_current_config()
print(f"Using configuration: {filename_suffix}")
if filename_suffix == "3" or filename_suffix == "4" or filename_suffix == "7" or filename_suffix == "8":
    raise Exception("Useless try")
iht = IHT(maxSize)
space_action_len = len(env.action_type.actions_indexes)

env.configure(config)
state, info = env.reset(seed=cu.get_seed())
np.random.seed(cu.get_seed())
random.seed(cu.get_seed())

weights_handler = WeightsHandler(maxSize, space_action_len)
weights = weights_handler.generate_weights()

avg_return = 0
avg_num_steps = 0
seed = cu.get_seed()

begin = datetime.datetime.now()
for episode in range(num_Episodes):
    run_time = round((datetime.datetime.now() - begin).total_seconds() / 60, 1)
    print("\nEpisode {}, avg_reward {:.3f}, avg_num_steps {:.3f}, seed {}, IHT usage: {}/{} ≈ {}%, run time: {} min".format(
        episode, avg_return, avg_num_steps, seed, iht.count(), iht.size, round(iht.count() / iht.size * 100, 2),
        run_time))
    done = truncated = False
    # Choose A and state S
    action = env.action_type.actions_indexes["IDLE"]
    state, info = env.reset(seed=seed)
    if episode == num_Episodes - 10:
        env = record_videos(env)
    # Debugging variables
    num_steps = 0
    expected_return = 0
    while not done and not truncated:
        # tiles_list of initial state
        tiles_list = tiles(iht, numTilings, state.flatten().tolist())
        # Take action A, observe R, S'
        state_p, reward, done, truncated, info = env.step(action)
        expected_return += reward
        if done or truncated:
            print("Episode finished after {} steps, crashed? {}, expected return {}".format(
                num_steps, done, expected_return))
        if done:
            for tile in tiles_list:
                weights[tile, action] = weights[tile, action] + alpha * (reward - estimate(tiles_list, action, weights))
        else:
            tiles_list_p = tiles(iht, numTilings, state_p.flatten().tolist())
            # Choose A' as a function of q(s, ., w) (e.g e-greedy)
            action_p = cu.get_e_greedy_action(epsilon, tiles_list_p, weights, random, env)
            for tile in tiles_list:
                weights[tile, action] = weights[tile, action] + alpha*(reward + gamma * estimate(tiles_list_p, action_p, weights) - estimate(tiles_list, action, weights))
            state = state_p
            action = action_p
            num_steps += 1
    seed += 1
    avg_num_steps += (num_steps - avg_num_steps) * 0.1
    avg_return += (expected_return - avg_return) * 0.1

print(f"IHT usage: {iht.count()}/{iht.size}")

os.makedirs("weights", exist_ok=True)
os.makedirs("ihts", exist_ok=True)
weights_handler.save_weights(weights, f"weights/episodic_semi_gradient_sarsa_weights{filename_suffix}")
su.serilizeIHT(iht, f"ihts/episodic_semi_gradient_sarsa_iht{filename_suffix}.pkl")
env.close()

