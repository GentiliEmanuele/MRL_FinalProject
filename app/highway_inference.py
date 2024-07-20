import configparser
import random
import warnings

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

import app.utilities.config_utils as cu

from app.tile_coding.my_tiles import IHT, tiles
from app.utilities.video_utils import record_videos
from app.utilities.weights_handler import WeightsHandler
from prettytable import PrettyTable

# Initialization
if True:
    # Suppress the specific warning message
    warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                    "wrappers is deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")

    # Create a ConfigParser object
    config_parser = configparser.ConfigParser()
    # Read the configuration file
    config_parser.read('config.ini')

    env = gym.make('highway-v0', render_mode='rgb_array')

    # Config the env
    config = cu.get_current_config()
    env.configure(config)

    # Reset seed
    np.random.seed(cu.get_seed())
    random.seed(cu.get_seed())

    maxSize = cu.get_max_size()
    numTilings = cu.get_num_tilings()
    iht = IHT(maxSize)

    space_action_len = len(env.action_type.actions_indexes)
    weights_handler = WeightsHandler(maxSize, space_action_len)

    special_type = ""
    model = None
    weights = None
    algorithm_type = int(config_parser['inference_algorithm']['type'])
    if algorithm_type == 1:
        print("Algorithm chosen: Episodic semi-gradient Sarsa")
        filename = "algorithms/weights/episodic_semi_gradient_sarsa_weights.npy"
        inference_name = "Episodic Semi Gradient SARSA"
    elif algorithm_type == 2:
        print("Algorithm chosen: True online TD(lambda)")
        filename = "algorithms/weights/true_online_td_lambda_weights.npy"
        inference_name = "True online TD Lambda"
    elif algorithm_type == 3:
        print("Algorithm chosen: Sarsa(lambda)")
        filename = "algorithms/weights/sarsa_lambda_weights.npy"
        inference_name = "Sarsa Lambda"
    elif algorithm_type == 4:
        print("Algorithm chosen: DQN")
        special_type = "DQN"
        filename = "algorithms/highway_dqn/model"
        inference_name = "DQN"
    else:
        print("Invalid configuration.\nAlgorithm chosen: Episodic semi-gradient Sarsa")
        raise Exception("Invalid name")

    if special_type == "DQN":
        model = DQN.load(filename)
    else:
        weights = weights_handler.load_weights(filename)
        if weights is None:
            print('Error in weights loading')
            exit(0)

# -------------------------------- INFERENCE BEGIN ----------------------------
inference_suffix = "Test inference"
inference_runs = 30

print_debug_each_step = False
print_debug_each_iteration = True
record_after_inference = 29

list_num_steps = np.zeros(inference_runs)
list_avg_speed = np.zeros(inference_runs)
list_avg_reward = np.zeros(inference_runs)
list_total_reward = np.zeros(inference_runs)

round_metrics = 3

for i in range(inference_runs):
    state, info = env.reset(seed=(42+i))

    done = False
    truncated = False
    num_steps = 0
    avg_speed = 0
    avg_reward = 0
    total_reward = 0

    if i == record_after_inference:
        env = record_videos(env)

    while not done and not truncated:
        # Get tilings for current state
        tiles_list = tiles(iht, numTilings, state.flatten().tolist())

        # Choose action
        if special_type == "DQN":
            action, _ = model.predict(state, deterministic=True)
        else:
            action = cu.get_e_greedy_action(0, space_action_len, tiles_list, weights, random)

        # Simulate
        state, reward, done, truncated, info = env.step(action)

        # Update
        num_steps += 1
        avg_speed += (1/num_steps)*(state[0][2] - avg_speed)
        avg_reward += (1/num_steps)*(reward - avg_reward)
        total_reward += reward

        if print_debug_each_step:
            status = cu.get_status_message(done, truncated)
            print("num_steps:{:03}\tspeed:{:.3f}\treward:{:.3f}\ttotal_reward:{:.3f}\tstatus:{}"
                  .format(num_steps, state[0][2], reward, total_reward, status))

    list_num_steps[i] = num_steps
    list_avg_speed[i] = avg_speed
    list_avg_reward[i] = avg_reward
    list_total_reward[i] = total_reward

    if print_debug_each_iteration:
        status = cu.get_status_message(done, truncated)
        print("Inference {:03} -> num_steps:{:03}\tav_speed:{:.3f}\tavg_reward:{:.3f}\ttotal_reward:{:.3f}\tstatus:{}"
              .format(i, num_steps, avg_speed, avg_reward, total_reward, status))

# Print name and info
print(f"\n\n{inference_name}-{inference_suffix}, maxSize:{maxSize}, numTilings:{numTilings}")

# Print mean and standard deviation of metrics
print("Measures mean and standard deviation:")
t = PrettyTable(['Measure', 'Mean', 'StdDev'])
mean_num_steps = round(np.mean(list_num_steps), round_metrics)
stddev_num_steps = round(np.std(list_num_steps), round_metrics)
t.add_row(["num_steps", mean_num_steps, stddev_num_steps])
mean_avg_speed = round(np.mean(list_avg_speed), round_metrics)
stddev_avg_speed = round(np.std(list_avg_speed), round_metrics)
t.add_row(["avg_speed", mean_avg_speed, stddev_avg_speed])
mean_avg_reward = round(np.mean(list_avg_reward), round_metrics)
stddev_avg_reward = round(np.std(list_avg_reward),round_metrics)
t.add_row(["avg_reward", mean_avg_reward, stddev_avg_reward])
mean_total_reward = round(np.mean(list_total_reward), round_metrics)
stddev_total_reward = round(np.std(list_total_reward), round_metrics)
t.add_row(["total_reward", mean_total_reward, stddev_total_reward])

print(t)

# Episodic Semi Gradient SARSA-Test inference, maxSize:12288, numTilings:24
# 1000 episodes, fixed seed
# +--------------+--------+--------+
# |   Measure    |  Mean  | StdDev |
# +--------------+--------+--------+
# |  num_steps   | 11.633 | 13.722 |
# |  avg_speed   | 0.311  | 0.032  |
# |  avg_reward  | 0.228  | 0.305  |
# | total_reward | 2.311  | 3.036  |
# +--------------+--------+--------+

# Episodic Semi Gradient SARSA-Test inference, maxSize:12288, numTilings:24
# 1000 episodes, seed changed once avg_expected_return > 20
# +--------------+--------+--------+
# |   Measure    |  Mean  | StdDev |
# +--------------+--------+--------+
# |  num_steps   | 12.267 | 10.856 |
# |  avg_speed   |  0.32  | 0.031  |
# |  avg_reward  | 0.423  | 0.249  |
# | total_reward | 4.458  | 4.444  |
# +--------------+--------+--------+

# Episodic Semi Gradient SARSA-Test inference, maxSize:12288, numTilings:24
# 1000 episodes, seed changing every iteration
# +--------------+-------+--------+
# |   Measure    |  Mean | StdDev |
# +--------------+-------+--------+
# |  num_steps   | 8.033 | 4.572  |
# |  avg_speed   | 0.341 | 0.025  |
# |  avg_reward  | 0.516 | 0.317  |
# | total_reward | 4.847 | 4.191  |
# +--------------+-------+--------+