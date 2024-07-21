import time
import warnings

import numpy as np
import app.utilities.state_utils as su
import gymnasium as gym

from tabulate import tabulate
from app.tile_coding.my_tiles import tiles, IHT
from app.utilities.config_utils import ConfigUtils

use_su = True

if True:
    # warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                    "wrappers is deprecated.*")

    cu = ConfigUtils()
    config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, _, num_Episodes = cu.get_current_config()

    config["observation"]["absolute"] = True
    config["observation"]["normalize"] = True
    # config["observation"]["features_range"] = {
    #         "dx": [-10, 90],
    #         "y": [-8, 8],
    #         "vx": [0, 30],
    #         "vy": [-5, 5]
    #     }

    features = cu.get_features()
    env = gym.make('highway-v0', render_mode='rgb_array')

    if use_su:
        su.custom_configure(env, config)
        state, info, nt_state = su.custom_reset(env, cu.get_seed())
    else:
        env.configure(config)
        state, info = env.reset(seed=cu.get_seed())
        nt_state = None

    iht = IHT(maxSize)
    available_action = env.action_type.get_available_actions()

old_tiles_list = None
done = truncated = False
while not (done or truncated):
    # Print state and tiles
    env.render()
    print("\n\n")

    if use_su:
        print(tabulate(nt_state, headers=features, tablefmt="grid"))
        tiles_list = tiles(iht, numTilings, state)
    else:
        print(tabulate(state, headers=features, tablefmt="grid"))
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
    action = available_action[0]

    if use_su:
        state, reward, done, truncated, info, nt_state = su.custom_step(env, action)
    else:
        state, reward, done, truncated, info = env.step(action)

    # Wait for user input to progress
    time.sleep(2)

    old_tiles_list = tiles_list
