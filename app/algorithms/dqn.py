import gymnasium as gym

from stable_baselines3 import DQN

from app.utilities.config_utils import ConfigUtils

cu = ConfigUtils()
config, _, _, _, _, _, _, _, _ = cu.get_current_config()

env = gym.make("highway-v0", render_mode='rgb_array')
env.configure(config)
state, info = env.reset(seed=cu.get_seed())

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
            verbose=2,
            tensorboard_log="highway_dqn/")

model.learn(int(1e4))
model.save("highway_dqn/model")

