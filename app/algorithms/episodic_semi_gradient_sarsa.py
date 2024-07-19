import configparser

import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.config_utils import get_features
from app.utilities.video_utils import record_videos
from matplotlib import pyplot as plt
import gymnasium as gym

import random

from app.utilities.weights_handler import WeightsHandler


class episodic_semi_gradient_sarsa():
    def execute(self, env, config):

        # Create a ConfigParser object
        config_parser = configparser.ConfigParser()
        # Read the configuration file
        config_parser.read('config.ini')

        # See initial configuration
        plt.imshow(env.render())
        plt.show()

        # optimal number of features (maxSize) per non fare hashing e vedere se facendo hashing le performance dell'algoritmo
        # degradano
        maxSize_proportion = int(config_parser['tilings']['maxSize_proportion'])
        maxSize = maxSize_proportion * config["observation"]["vehicles_count"] * len(get_features())
        iht = IHT(maxSize)
        # according to Sutton example we keep the ratio between maxSize and numTilings as 1 / 256
        numTilings = maxSize // maxSize_proportion

        space_action_len = len(env.action_type.actions_indexes)
        weights_handler = WeightsHandler(maxSize, space_action_len)
        weights = weights_handler.generate_weights()

        alpha = 0.1 / numTilings  # step size
        epsilon_0 = 0.1
        epsilon = epsilon_0
        gamma = 0.9
        num_Episodes = 201

        # Choose A
        for episode in range(num_Episodes):
            print("Episode", episode)
            done = False
            truncated = False
            if episode % 50 == 0 and episode != 0:
                env = record_videos(env)

            if episode % 50 == 1:
                env = gym.make('highway-v0', render_mode='rgb_array')
                env.configure(config)

            state, info = env.reset(seed=42 + episode)
            action = env.action_type.actions_indexes["IDLE"]

            # epsilon = epsilon - epsilon_0 / num_Episodes

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
                    for tile in tiles_list:
                        weights[tile, action] = weights[tile, action] + alpha * (
                                reward - estimate(tiles_list, action, weights))
                else:
                    tiles_list_p = tiles(iht, numTilings, state_p.flatten().tolist())
                    # Choose A' as a function of q(s, ., w) (e.g e-greedy)
                    if random.random() < epsilon:
                        action_p = random.randint(0, space_action_len - 1)
                    else:
                        best_action = 0
                        best_estimate = estimate(tiles_list_p, 0, weights)
                        for a in range(1, space_action_len):
                            actual_estimate = estimate(tiles_list_p, a, weights)
                            if actual_estimate > best_estimate:
                                best_estimate = actual_estimate
                                best_action = a
                        action_p = best_action
                    for tile in tiles_list:
                        weights[tile, action] = weights[tile, action] + alpha * (
                                reward + gamma * estimate(tiles_list_p, action_p, weights) - estimate(tiles_list,
                                                                                                      action,
                                                                                                      weights))
                    state = state_p
                    action = action_p
                    num_steps += 1

        #-----------------------INFERENCE--------------------------
        inference = True
        if inference:
            print("STO QUA")
            env.close()
            env = gym.make('highway-v0', render_mode='rgb_array')
            env.configure(config)
            state, info = env.reset(seed=42)
            np.random.seed(44)
            random.seed(44)

            features = ["x", "y", "vx", "vy"]
            maxSize_proportion = int(config_parser['tilings']['maxSize_proportion'])
            maxSize = maxSize_proportion * config["observation"]["vehicles_count"] * len(features)
            iht = IHT(maxSize)
            numTilings = maxSize // maxSize_proportion

            space_action_len = len(env.action_type.actions_indexes)

            avg_avg_speed = 0
            avg_num_steps = 0
            inference_runs = 10

            for i in range(10):
                state, info = env.reset(seed=(42+i))
                action = env.action_type.actions_indexes["IDLE"]

                done = False
                truncated = False
                avg_speed = 0
                num_steps = 0
                env.configure(config)
                if False and i == 0:
                    env = record_videos(env)
                while not done and not truncated:
                    tiles_list = tiles(iht, numTilings, state.flatten().tolist())

                    best_action = 0
                    best_estimate = estimate(tiles_list, 0, weights)
                    for a in range(1, space_action_len):
                        actual_estimate = estimate(tiles_list, a, weights)
                        if actual_estimate > best_estimate:
                            best_estimate = actual_estimate
                            best_action = a
                    action = best_action

                    num_steps += 1
                    avg_speed = ((num_steps-1)*avg_speed + state[0][2]) / num_steps

                    state, reward, done, truncated, info = env.step(action)
                avg_avg_speed += avg_speed / inference_runs
                avg_num_steps += num_steps / inference_runs
                print(f"Inference {i} -> avg_speed: {avg_speed}, num_steps: {num_steps}")

            print("Average avg_speed: {}, Average num_steps={}".format(avg_avg_speed, avg_num_steps))
            #-----------------------------END INFERENCE-------------------------------


        # ---------------------------- WEIGHTS -----------------------------------
        weights_handler.save_weights(weights, "algorithms/weights/episodic_semi_gradient_sarsa_weights")

        saved_weights = weights_handler.load_weights("algorithms/weights/episodic_semi_gradient_sarsa_weights.npy")

        print(f"Equals? {np.array_equal(weights, saved_weights)}")

        return weights
