import datetime
import random
import gymnasium as gym
import numpy as np
from prettytable import PrettyTable

from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.config_utils import ConfigUtils
from app.utilities.video_utils import record_videos
import warnings

import app.utilities.serialization_utils as su
import app.utilities.state_utils as stu

from app.utilities.weights_handler import WeightsHandler

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")

env = gym.make('highway-v0', render_mode='rgb_array')

cu = ConfigUtils()
config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, _, num_Episodes = cu.get_current_config()
iht = IHT(maxSize)
space_action_len = len(env.action_type.actions_indexes)

stu.custom_configure(env, config)
state, info, _ = stu.custom_reset(env, cu.get_seed())
np.random.seed(cu.get_seed())
random.seed(cu.get_seed())

done = False
truncated = False

weights_handler = WeightsHandler(maxSize, space_action_len)
weights = weights_handler.generate_weights()

avg_return = 0
avg_num_steps = 0
seed = cu.get_seed()

print(f"ID: {filename_suffix}")
begin = datetime.datetime.now()
for episode in range(num_Episodes):
    run_time = round((datetime.datetime.now() - begin).total_seconds() / 60, 1)
    print("\nEpisode {}, avg_reward {:.3f}, avg_num_steps {:.3f}, seed {}, IHT usage: {}/{} ≈ {}%, run time: {} min".format(
        episode, avg_return, avg_num_steps, seed, iht.count(), iht.size, round(iht.count()/iht.size*100, 2), run_time))
    done = truncated = False
    # Choose A and state S
    action = env.action_type.actions_indexes["IDLE"]
    state, info, _ = stu.custom_reset(env, cu.get_seed())
    if episode == num_Episodes - 10:
        env = record_videos(env)
    # Debugging variables
    num_steps = 0
    expected_return = 0
    while not done and not truncated:
        # tiles_list of initial state
        tiles_list = tiles(iht, numTilings, state)
        estimate_value = estimate(tiles_list, action, weights)
        # Take action A, observe R, S'
        state_p, reward, done, truncated, info, _ = stu.custom_step(env, action)
        expected_return += reward
        if done or truncated:
            print("Finished after {} steps, expected return {:.3f}, crashed? {}".format(
                num_steps, expected_return, done))
        if done:
            for tile in tiles_list:
                weights[tile, action] = weights[tile, action] + alpha * (reward - estimate_value)
        else:
            tiles_list_p = tiles(iht, numTilings, state_p)
            # Choose A' as a function of q(s, ., w) (e.g e-greedy)
            action_p = cu.get_e_greedy_action(epsilon, tiles_list_p, weights, random, env)
            estimate_value_p = estimate(tiles_list_p, action_p, weights)
            for tile in tiles_list:
                weights[tile, action] = weights[tile, action] + alpha * (
                        reward + gamma * estimate_value_p - estimate_value)
            state = state_p
            action = action_p
            num_steps += 1
    avg_num_steps += (num_steps - avg_num_steps) * 0.125
    avg_return += (expected_return - avg_return) * 0.125
    seed += 1


weights_handler.save_weights(weights, f"weights/episodic_semi_gradient_sarsa_weights{filename_suffix}")
su.serilizeIHT(iht, f"ihts/episodic_semi_gradient_sarsa_iht{filename_suffix}.pkl")
env.close()

# ------------------------- INFERENCE -------------------------------
if True:
    env = gym.make('highway-v0', render_mode='rgb_array')

    # Config the env
    config, _, _, _ = cu.get_inference_config()
    stu.custom_configure(env, config)

    # Reset seed
    np.random.seed(cu.get_seed())
    random.seed(cu.get_seed())

    print("Algorithm chosen: Episodic semi-gradient Sarsa")
    inference_name = "Episodic Semi Gradient SARSA"

    inference_suffix = "Test inference"
    inference_runs = 50

    print_debug_each_step = False
    print_debug_each_iteration = True
    record_after_inference = 50

    list_num_steps = np.zeros(inference_runs)
    list_avg_speed = np.zeros(inference_runs)
    list_avg_reward = np.zeros(inference_runs)
    list_total_reward = np.zeros(inference_runs)

    round_metrics = 3

    for i in range(inference_runs):
        state, info, _ = stu.custom_reset(env, cu.get_seed() + i)

        done = False
        truncated = False
        num_steps = 0
        avg_speed = 0.0
        avg_reward = 0.0
        total_reward = 0.0

        if i == record_after_inference:
            env = record_videos(env)

        while not done and not truncated:
            # Get tilings for current state
            tiles_list = tiles(iht, numTilings, state)

            # Choose action
            action = cu.get_e_greedy_action(-1, tiles_list, weights, random, env)

            # Simulate
            state, reward, done, truncated, info, _ = stu.custom_step(env, action)

            # Update
            num_steps += 1
            avg_speed += (1 / num_steps) * (state[1] - avg_speed)
            avg_reward += (1 / num_steps) * (reward - avg_reward)
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
            print(
                "Inference {:03} -> num_steps:{:03}\tav_speed:{:.3f}\tavg_reward:{:.3f}\ttotal_reward:{:.3f}\tstatus:{}"
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
    stddev_avg_reward = round(np.std(list_avg_reward), round_metrics)
    t.add_row(["avg_reward", mean_avg_reward, stddev_avg_reward])
    mean_total_reward = round(np.mean(list_total_reward), round_metrics)
    stddev_total_reward = round(np.std(list_total_reward), round_metrics)
    t.add_row(["total_reward", mean_total_reward, stddev_total_reward])

    print(t)

saved_weights = weights_handler.load_weights(f"weights/episodic_semi_gradient_sarsa_weights{filename_suffix}.npy")
print(f"Equals weights? {weights == saved_weights}")
print(f"Sum: {np.sum(weights - saved_weights)}")
