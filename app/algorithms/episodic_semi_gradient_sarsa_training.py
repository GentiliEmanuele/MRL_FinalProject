import random
import configparser
import gymnasium as gym
import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.video_utils import record_videos
import warnings
from matplotlib import pyplot as plt
import app.utilities.config_utils as cu

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
numTilings = cu.get_num_tilings() # according to Sutton example we keep the ratio between maxSize and numTilings as 1 / 156
alpha = cu.get_alpha()# step size
epsilon_0 = cu.get_epsilon0()
epsilon = epsilon_0
gamma = cu.get_gamma()
num_Episodes = 1000

weights_handler = WeightsHandler(maxSize, space_action_len)
weights = weights_handler.generate_weights()

# Create a ConfigParser object
config_parser = configparser.ConfigParser()
# Read the configuration file
config_parser.read('config.ini')

avg_return = 0
seed_episodes = 0
seed = cu.get_seed()
for episode in range(num_Episodes):
    print(f"#episodes {episode}, avg_reward {avg_return}, seed {seed}")
    done = False
    truncated = False
    # Choose A and state S
    action = env.action_type.actions_indexes["IDLE"]
    state, info = env.reset(seed=seed)
    if episode == num_Episodes - 20:
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
            print("Episode finished after {} timesteps, crashed? {}".format(num_steps, done))
            print("Expected return {}".format(expected_return))
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
    seed_episodes += 1
    avg_return += (expected_return - avg_return) * 0.2
    if True or avg_return > -1:
        seed_episodes = 0
        # avg_return = 0
        seed = seed + 1
        # print(f"change seed {seed}")

print(f"IHT usage: {iht.count()}/{iht.size}")
weights_handler.save_weights(weights, "weights/episodic_semi_gradient_sarsa_weights")
env.close()

