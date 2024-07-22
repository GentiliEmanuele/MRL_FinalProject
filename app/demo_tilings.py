import time
import warnings
from random import random, randint

import numpy as np
from tabulate import tabulate

from app.utilities.config_utils import ConfigUtils

import gymnasium as gym
from matplotlib import pyplot as plt

from app.tile_coding.my_tiles import tiles, IHT

if True:
    warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                    "wrappers is deprecated.*")

    cu = ConfigUtils()
    config, filename_suffix, maxSize, numTilings = cu.get_inference_config()
    config["observation"]["absolute"] = False
    config["observation"]["normalize"] = False
    features = cu.get_features()
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.configure(config)
    state, info = env.reset(seed=42)

    iht = IHT(maxSize)
    available_action = env.action_type.get_available_actions()


old_tiles_list = None
done = truncated = False
while not (done or truncated):
    # Print state and tiles
    env.render()
    print("\n\n")
    print(tabulate(state, headers=features, tablefmt="grid"))

    real_speed = state[0][2] / 0.0125
    print(f"real_speed: {real_speed}")

    tiles_list = tiles(iht, numTilings, state.flatten().tolist())
    print(f"Active tiles:{tiles_list}")

    if old_tiles_list is not None:
        difference_sum = np.setdiff1d(tiles_list, old_tiles_list)
        total_tiles = len(tiles_list)
        changed_tiles = len(difference_sum)
        print("Difference tiles: {}/{} ≈ {}%".format(
            changed_tiles,
            total_tiles,
            round(changed_tiles / total_tiles * 100, 1)))

    # Choose and take action
    # action_index = randint(0, 4)
    # action = available_action[action_index]
    action = env.action_type.actions_indexes["FASTER"]
    state, reward, done, truncated, info = env.step(action)

    # Wait for user input to progress
    time.sleep(2)

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
