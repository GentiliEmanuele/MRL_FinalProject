import time
import warnings

import numpy as np
from tabulate import tabulate

import app.utilities.config_utils as cu

import gymnasium as gym
from matplotlib import pyplot as plt

from app.tile_coding.my_tiles import tiles, IHT

if True:
    warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                    "wrappers is deprecated.*")

    config = cu.get_current_config()
    features = cu.get_features()
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.configure(config)
    state, info = env.reset()

    maxSize = cu.get_max_size()
    iht = IHT(maxSize)
    numTilings = cu.get_num_tilings()

old_tiles_list = None
for _ in range(3):
    # Print state and tiles
    env.render()
    print(tabulate(state, headers=features, tablefmt="grid"))

    tiles_list = tiles(iht, numTilings, state.flatten().tolist())
    np.sort(tiles_list)
    print(tiles_list)

    if old_tiles_list is not None:
        difference_array = old_tiles_list != tiles_list
        difference_sum = np.sum(difference_array)
        print("Difference tiles: {}".format(difference_sum))

    # Choose and take action
    action = env.action_type.actions_indexes["IDLE"]
    state, reward, done, truncated, info = env.step(action)

    # Wait for user input to progress
    input()

    old_tiles_list = tiles_list

# plt.imshow(env.render())
# plt.show()

# TRAINING
# s = [0.5, 0.6, 0.7]
# 1 2 3 4
# 5 10 -1 0
# estimate(s) = 5*1 + 10*2 + -1*3 + 0*4

# s = [0.8, 0.9, 1.0]
# 1 2 3 5
# 5 10 -1 30
# estimate(s) = 5*1 + 10*2 + -1*3 + 30*5

# INFERENCE
# s = [31.5, 456.7, 2000.4]
# 1 2 3 4
# 5 10 -1 0
# estimate(s) = 5*1 + 10*2 + -1*3 + 0*4
