"""
Sutton page 244
"""
from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.video_utils import record_videos
from matplotlib import pyplot as plt

import numpy as np
import random

from app.utilities.weights_handler import WeightsHandler


class true_online_td_lambda():
    def execute(self, env, config):
        # See initial configuration
        plt.imshow(env.render())
        plt.show()

        features = ["x", "y", "vx", "vy"]

        done = False
        truncated = False
        # optimal number of features (maxSize) per non fare hashing e vedere se facendo hashing le performance dell'algoritmo
        # degradano
        maxSize = 256 * 12
        iht = IHT(maxSize)
        # according to Sutton example we keep the ratio between maxSize and numTilings as 1 / 256
        numTilings = maxSize // 256

        space_action_len = len(env.action_type.actions_indexes)
        weights_handler = WeightsHandler(maxSize, space_action_len)
        weights = weights_handler.generate_weights()

        alpha = 0.1 / numTilings  # step size
        epsilon_0 = 0.1
        epsilon = epsilon_0
        gamma = 0.9
        trace_decay_rate = 0.9  #lambda
        num_Episodes = 100

        for episode in range(num_Episodes):
            print("Episode", episode)
            done = False
            truncated = False
            if episode == num_Episodes - 1:
                config["duration"] = 160
                config["vehicles_count"] = 20
                env.configure(config)
                env = record_videos(env)

            state, info = env.reset(seed=42)

            # x
            tiles_list = tiles(iht, numTilings, state.flatten().tolist())
            # z
            traces = np.zeros(weights.shape)

            V_old = 0

            # Loop for each step of episode
            while not done and not truncated:
                # Choose A
                if random.random() < epsilon:
                    action = random.randint(0, space_action_len - 1)
                else:
                    best_action = 0
                    best_estimate = 0
                    for a in range(0, space_action_len):
                        actual_estimate = estimate(tiles_list, a, weights)
                        if actual_estimate > best_estimate:
                            best_estimate = actual_estimate
                            best_action = a
                    action = best_action

                # Take action A, observe R, S'
                state_p, reward, done, truncated, info = env.step(action)

                if done:
                    break

                # x'
                tiles_list_p = tiles(iht, numTilings, state_p.flatten().tolist())

                V = 0.0
                V_p = 0.0

                for tile in tiles_list:
                    V = V + weights[tile, action]

                for tile_p in tiles_list_p:
                    V_p = V_p + weights[tile_p, action]

                delta = reward + gamma * V_p - V

                for tile in tiles_list:
                    traces[tile, action] = (gamma * trace_decay_rate * traces[tile, action] +
                          (1 - alpha * gamma * trace_decay_rate * traces[tile, action]))

                weights = weights + alpha * (delta + V - V_p) * traces
                for tile in tiles_list:
                    weights[tile, action] = weights[tile, action] - alpha * (V - V_p)

                V_old = V_p

                tiles_list = tiles_list_p

                epsilon = epsilon - epsilon_0 / num_Episodes

        weights_handler.save_weights(weights, "algorithms/weights/true_online_td_lambda_weights")

        return weights
