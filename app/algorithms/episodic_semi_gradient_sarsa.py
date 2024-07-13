"""
Sutton page 244
"""
from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.video_utils import record_videos
from matplotlib import pyplot as plt

import random

from app.utilities.weights_handler import WeightsHandler


class episodic_semi_gradient_sarsa():
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
        num_Episodes = 200

        # Choose A
        action = env.action_type.actions_indexes["IDLE"]
        for episode in range(num_Episodes):
            print("Episode", episode)
            done = False
            truncated = False
            state, info = env.reset(seed=44)
            if episode == num_Episodes - 1:
                config["duration"] = 160
                env.configure(config)
                env = record_videos(env)
            while not done and not truncated:
                # tiles_list of initial state
                tiles_list = tiles(iht, numTilings, state.flatten().tolist())
                # Take action A, observe R, S'
                state_p, reward, done, truncated, info = env.step(action)
                if done:
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
                        best_estimate = 0
                        for a in range(0, space_action_len):
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
                epsilon = epsilon - epsilon_0 / num_Episodes

        weights_handler.save_weights(weights, "algorithms/weights/episodic_semi_gradient_sarsa_weights")

        return weights
