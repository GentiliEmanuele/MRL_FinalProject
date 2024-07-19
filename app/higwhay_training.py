import random

import gymnasium as gym
import numpy as np

import warnings
import configparser

from app.algorithms.episodic_semi_gradient_sarsa import episodic_semi_gradient_sarsa
from app.algorithms.true_online_td_lambda import true_online_td_lambda
from app.utilities.config_utils import get_current_config

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other "
                                                                "wrappers is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")


# Create a ConfigParser object
config_parser = configparser.ConfigParser()
# Read the configuration file
config_parser.read('config.ini')

config = get_current_config()

env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(config)
state, info = env.reset(seed=44)
np.random.seed(44)
random.seed(44)

algorithm_type = int(config_parser['algorithm']['type'])
if algorithm_type == 1:
    print("Algorithm chosen: Episodic semi-gradient Sarsa")
    algorithm_instance = episodic_semi_gradient_sarsa()
elif algorithm_type == 2:
    print("Algorithm chosen: True online TD(lambda)")
    algorithm_instance = true_online_td_lambda()
else:
    print("Invalid configuration.\nAlgorithm chosen: Episodic semi-gradient Sarsa")
    algorithm_instance = episodic_semi_gradient_sarsa()

weights = algorithm_instance.execute(env=env, config=config)

env.close()
