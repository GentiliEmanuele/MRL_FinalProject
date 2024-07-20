from random import random

from app.tile_coding.my_tiles import estimate


def get_current_config():
    config = {
        "observation": {
            "type": "Kinematics",
            "features": get_features(),
            "absolute": False,
            "order": "sorted",
            "vehicles_count": 4,  # max number of observable vehicles
            "normalize": True
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 10,  # max number of existing vehicles
        "duration": 72,  # [s]
        "initial_spacing": 2,
        "collision_reward": -1,  # The reward received when colliding with a vehicle.
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
    return config


def get_features():
    return ["x", "y", "vx", "vy"]


def get_max_size():
    return 1024 * len(get_features())


def get_num_tilings():
    return get_max_size() // 512


def get_alpha():
    return 0.1 / get_num_tilings()


def get_epsilon0():
    return 0.1


def get_gamma():
    return 0.9


def get_lambda():
    return 0.9


def get_seed():
    return 44

def get_e_greedy_action(epsilon, space_action_len, tiles_list, weights):
    if random.random() < epsilon:
        action = random.randint(0, space_action_len - 1)
    else:
        best_action = 0
        best_estimate = estimate(tiles_list, 0, weights)
        for a in range(1, space_action_len):
            actual_estimate = estimate(tiles_list, a, weights)
            if actual_estimate > best_estimate:
                best_estimate = actual_estimate
                best_action = a
        action = best_action