import gymnasium as gym
import app.utilities.config_utils as cu
import datetime

from stable_baselines3 import DQN

from app.utilities.video_utils import record_videos

config = cu.get_current_config()

env = gym.make("highway-v0", render_mode='rgb_array')
env.configure(config)
state, info = env.reset(seed=cu.get_seed())

print("Before training: ", datetime.datetime.now())

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
            verbose=0,
            tensorboard_log="highway_dqn/")

model.learn(int(1e4))
model.save("highway_dqn/model")

print("After training: ", datetime.datetime.now())

# Load and test saved model
model = DQN.load("highway_dqn/model")
env = record_videos(env)
for i in range(1):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
