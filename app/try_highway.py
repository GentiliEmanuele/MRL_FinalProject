import random
import configparser
import gymnasium as gym
import numpy as np

from app.tile_coding.my_tiles import IHT, tiles, estimate
from app.utilities.video_utils import record_videos
import warnings
from matplotlib import pyplot as plt

from app.utilities.weights_handler import WeightsHandler

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")

config = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "absolute": False,
        "order": "sorted",
        "vehicles_count": 4, #max number of observable vehicles
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 20, #max number of existing vehicles
    "duration": 72,  # [s]
    "initial_spacing": 2,
    "collision_reward": -50,  # The reward received when colliding with a vehicle.
    "high_speed_reward": 0.3,
    "reward_speed_range": [25, 30], # [m/s] The reward for high speed is mapped linearly from this range to [0,
    # HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 1200,  # [px]
    "screen_height": 250,  # [px]
    "centering_position": [0.1, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": False,
    "offscreen_rendering": False
}

env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)
state, info = env.reset(seed=44)
np.random.seed(44)
random.seed(44)

# See initial configuration
plt.imshow(env.render())
plt.show()
#env = record_videos(env)


features = ["x", "y", "vx", "vy"]

done = False
truncated = False

maxSize = 1024 * 12
iht = IHT(maxSize)
space_action_len = len(env.action_type.actions_indexes)
# weights = np.random.rand(maxSize, space_action_len)
numTilings = maxSize // 512 # according to Sutton example we keep the ratio between maxSize and numTilings as 1 / 156
alpha = 0.1 / numTilings # step size
epsilon_0 = 0.1
epsilon = epsilon_0
gamma = 0.9
num_Episodes = 2000

weights_handler = WeightsHandler(maxSize, space_action_len)
weights = weights_handler.generate_weights()

# Create a ConfigParser object
config_parser = configparser.ConfigParser()
# Read the configuration file
config_parser.read('config.ini')

for episode in range(num_Episodes):
    print("Episode", episode)
    done = False
    truncated = False
    # Choose A and state S
    action = env.action_type.actions_indexes["IDLE"]
    state, info = env.reset(seed=44 + episode)
    if episode == num_Episodes - 1:
        config["duration"] = 80
        env.configure(config)
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
            if random.random() < epsilon:
                action_p = random.randint(0, space_action_len - 1)
            else:
                best_action = 0
                best_estimate = estimate(tiles_list_p, 0, weights)
                for a in range(0, space_action_len):
                    actual_estimate = estimate(tiles_list_p, a, weights)
                    if actual_estimate > best_estimate:
                        best_estimate = actual_estimate
                        best_action = a
                action_p = best_action
            for tile in tiles_list:
                weights[tile, action] = weights[tile, action] + alpha*(reward + gamma * estimate(tiles_list_p, action_p, weights) - estimate(tiles_list, action, weights))
            state = state_p
            action = action_p
            num_steps += 1
        # epsilon = epsilon - epsilon_0 / num_Episodes

# weights_handler.save_weights(weights, "algorithms/weights/episodic_semi_gradient_sarsa_weights")
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


print(weights)
env.close()