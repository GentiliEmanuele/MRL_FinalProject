import gymnasium as gym
from video_utils import record_videos
import warnings
from tabulate import tabulate

# Suppress the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, message=".*env.action_type to get variables from other wrappers is deprecated.*")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*env.configure.*")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*Overwriting existing videos.*")

config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": True,
        "order": "sorted",
        "vehicles_count": 5, # max number of observable vehicles
        "normalize": False
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 5,
    "vehicles_count": 10, # max number of existing vehicles
    "duration": 40,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
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

print("Config done")

env = gym.make('highway-v0', render_mode='rgb_array')

print("Gym make")

env.configure(config)

print("Gym configure")

obs, info = env.reset(seed = 666)

features = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
print(tabulate(obs, headers = features, tablefmt="grid"))

env = record_videos(env)

done = False
truncated = False


while (not done and not truncated):
    action = env.action_type.actions_indexes["FASTER"]
    obs, reward, done, truncated, info = env.step(action)

env.close()
