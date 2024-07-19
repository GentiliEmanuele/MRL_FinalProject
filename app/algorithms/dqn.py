import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from app.utilities.video_utils import record_videos

features = ["x", "y", "vx", "vy"]
config = {
    "observation": {
        "type": "Kinematics",
        "features": features,
        "absolute": False,
        "order": "sorted",
        "vehicles_count": 4, #max number of observable vehicles
        "normalize": True
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 10, # max number of existing vehicles
    "duration": 72,  # [s]
    "initial_spacing": 2,
    "collision_reward": -50,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30], # [m/s] The reward for high speed is mapped linearly from this range to [0,
    # HighwayEnv.HIGH_SPEED_REWARD].
    "high_speed_reward": 0.3,
    "normalize_reward": True,
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

env = gym.make("highway-v0", render_mode='rgb_array')
# plt.imshow(env.render())
# plt.show()
env.configure(config)
state, info = env.reset(seed=44)

model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")
model.learn(int(2e4))
model.save("highway_dqn/model")

# Load and test saved model
model = DQN.load("highway_dqn/model")
env = record_videos(env)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    # env.render()