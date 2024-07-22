import configparser

from app.tile_coding.my_tiles import estimate


class ConfigUtils:
    def get_current_config(self):
        # Create a ConfigParser object
        config_parser = configparser.ConfigParser()
        # Read the configuration file
        try:
            config_parser.read('../config.ini')
        except:
            raise Exception("Error in reading config.ini file: check that:\n"
                            "1) You didn't move config_utils.py in a different path from app/utilities\n"
                            "2) You didn't moved config.ini in a different path from app/")

        n = int(config_parser['configuration_info']['id'])
        method_name = f'get_current_config{n}'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method()
        else:
            print(f"No method found for get_current_config{n}")
            raise Exception("Invalid config i")

    def get_inference_config(self):
        # Create a ConfigParser object
        config_parser = configparser.ConfigParser()
        # Read the configuration file
        try:
            config_parser.read('../config.ini')
        except:
            raise Exception("Error in reading config.ini file: check that:\n"
                            "1) You didn't move config_utils.py in a different path from app/utilities\n"
                            "2) You didn't moved config.ini in a different path from app/")

        n = config_parser['configuration_info']['id']
        method_name = f'get_inference_config{n}'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method()
        else:
            print(f"No method found for get_current_config{n}")
            raise Exception("Invalid config i")

    def get_current_config1(self):
        filename_suffix = "1"
        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": False,
                "order": "sorted",
                "vehicles_count": 4,  # max number of observable vehicles
                "normalize": True
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 20,  # max number of existing vehicles
            "duration": 72,  # [s]
            "initial_spacing": 2,
            "collision_reward": -10,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [25, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0,
            # HighwayEnv.HIGH_SPEED_REWARD].
            "high_speed_reward": 1,
            "normalize_reward": False,
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

        maxSize = 1024 * len(self.get_features()) * 3
        numTilings = maxSize // 128
        alpha = 0.1 / numTilings
        epsilon_0 = 0.1
        epsilon = epsilon_0
        gamma = 0.9
        lambda_ = 0.9
        num_Episodes = 1000

        return config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, lambda_, num_Episodes

    def get_current_config2(self):
        filename_suffix = "2"
        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": True,
                "order": "sorted",
                "vehicles_count": 4,  # max number of observable vehicles
                "normalize": True
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 20,  # max number of existing vehicles
            "duration": 72,  # [s]
            "initial_spacing": 2,
            "collision_reward": -10,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [25, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0,
            # HighwayEnv.HIGH_SPEED_REWARD].
            "high_speed_reward": 1,
            "normalize_reward": False,
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

        maxSize = 1024 * len(self.get_features()) * 3
        numTilings = maxSize // 128
        alpha = 0.1 / numTilings
        epsilon_0 = 0.1
        epsilon = epsilon_0
        gamma = 0.9
        lambda_ = 0.9
        num_Episodes = 1000

        return config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, lambda_, num_Episodes

    def get_current_config3(self):
        filename_suffix = "3"
        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": False,
                "order": "sorted",
                "vehicles_count": 4,  # max number of observable vehicles
                "normalize": True,
                "normalize_scale": [0, 1]
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 20,  # max number of existing vehicles
            "duration": 72,  # [s]
            "initial_spacing": 2,
            "collision_reward": -10,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [25, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0,
            # HighwayEnv.HIGH_SPEED_REWARD].
            "high_speed_reward": 1,
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

        maxSize = 1024 * 10 * len(self.get_features()) * 4
        numTilings = 96
        alpha = 0.1 / numTilings
        epsilon_0 = 0.1
        epsilon = epsilon_0
        gamma = 0.9
        lambda_ = 0.9
        num_Episodes = 1000

        return config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, lambda_, num_Episodes

    def get_current_config4(self):
        filename_suffix = "4"
        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": False,
                "order": "sorted",
                "vehicles_count": 4,  # max number of observable vehicles
                "normalize": True,
                # "normalize_scale": [0, 0],
                # "features_range": {
                #     "dx": [-5, 50]
                # }
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 20,  # max number of existing vehicles
            "duration": 60,  # [s]
            "initial_spacing": 2,
            "collision_reward": -10,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [25, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0,
            # HighwayEnv.HIGH_SPEED_REWARD].
            "high_speed_reward": 1,
            "normalize_reward": False,
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

        maxSize = 1024 * 16 * len(self.get_features()) * 4
        numTilings = 128
        alpha = 0.1 / numTilings
        epsilon_0 = 0.1
        epsilon = epsilon_0
        gamma = 0.9
        lambda_ = 0.9
        num_Episodes = 1000

        return config, filename_suffix, maxSize, numTilings, alpha, epsilon, gamma, lambda_, num_Episodes


    def get_inference_config1(self):
        config, filename_suffix, maxSize, numTilings, _, _, _, _, _ = self.get_current_config()
        config["normalize_reward"] = False
        config["collision_reward"] = -1
        return config, filename_suffix, maxSize, numTilings

    def get_inference_config2(self):
        config, filename_suffix, maxSize, numTilings, _, _, _, _, _ = self.get_current_config()
        config["normalize_reward"] = False
        config["collision_reward"] = -1
        return config, filename_suffix, maxSize, numTilings

    def get_inference_config3(self):
        config, filename_suffix, maxSize, numTilings, _, _, _, _, _ = self.get_current_config()
        config["normalize_reward"] = False
        config["collision_reward"] = -1
        return config, filename_suffix, maxSize, numTilings

    def get_inference_config4(self):
        config, filename_suffix, maxSize, numTilings, _, _, _, _, _ = self.get_current_config()
        config["normalize_reward"] = False
        config["collision_reward"] = -1
        return config, filename_suffix, maxSize, numTilings

    def get_seed(self):
        return 44

    def get_features(self):
        return ["x", "y", "vx", "vy"]

    def get_e_greedy_action(self, epsilon, tiles_list, weights, random, env):
        available_action = env.action_type.get_available_actions()
        if random.random() < epsilon:
            action_index = random.randint(0, len(available_action) - 1)
            action = available_action[action_index]
        else:
            best_action = available_action[0]
            best_estimate = estimate(tiles_list, available_action[0], weights)
            for a in available_action[1:]:
                actual_estimate = estimate(tiles_list, a, weights)
                if actual_estimate > best_estimate:
                    best_estimate = actual_estimate
                    best_action = a
            action = best_action

        return action

    def get_GLIE_action(self, epsilon, tiles_list, weights, random, episode, num_episodes, env):
        available_action = env.action_type.get_available_actions()
        if random.random() < epsilon * (1 - episode / num_episodes):
            action_index = random.randint(0, len(available_action) - 1)
            action = available_action[action_index]
        else:
            best_action = available_action[0]
            best_estimate = estimate(tiles_list, available_action[0], weights)
            for a in available_action[1:]:
                actual_estimate = estimate(tiles_list, a, weights)
                if actual_estimate > best_estimate:
                    best_estimate = actual_estimate
                    best_action = a
            action = best_action

        return action

    def get_status_message(self, done, truncated):
        if done:
            return "Crashed"
        elif truncated:
            return "Truncated"
        else:
            return "On going"
