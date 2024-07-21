import numpy as np


# env.configure(config)
def custom_configure(env, config):
    try:
        normalize = config["observation"]["normalize"]
    except KeyError:
        normalize = True
    try:
        absolute = config["observation"]["absolute"]
    except KeyError:
        absolute = False

    t_config = config.copy()
    if absolute:
        t_config["observation"]["normalize"] = False
        t_config["observation"]["custom_normalize"] = normalize
    env.configure(t_config)


# state, info = env.reset(seed=cu.get_seed())
def custom_reset(env, custom_seed):
    state, info = env.reset(seed=custom_seed)
    t_state, nt_state = transform_state(env, state)
    return t_state, info, nt_state


# state, reward, done, truncated, info = env.step(action)
def custom_step(env, action):
    state, reward, done, truncated, info = env.step(action)
    t_state, nt_state = transform_state(env, state)
    return t_state, reward, done, truncated, info, nt_state


def transform_state(env, state):
    try:
        is_absolute = env.unwrapped.config["observation"]["absolute"]
    except KeyError:
        is_absolute = False
    if is_absolute:
        agent_x = state[0][0]
        state[:, 0] -= agent_x
        if env.unwrapped.config["observation"]["custom_normalize"]:
            state = normalize_state(env, state)
    nt_state = state
    return state.flatten()[1:].tolist(), nt_state


def normalize_state(env, state):
    src_dx, src_y, src_vx, src_vy = get_bounds(env)
    state[:, 0] = np.interp(state[:, 0], src_dx, [0, 1])
    state[:, 1] = np.interp(state[:, 1], src_y, [0, 1])
    state[:, 2] = np.interp(state[:, 2], src_vx, [0, 1])
    state[:, 3] = np.interp(state[:, 3], src_vy, [0, 1])
    return state


def get_bounds(env):
    try:
        src_dx = env.unwrapped.config["observation"]["features_range"]["dx"]
    except KeyError:
        src_dx = [-10, 90]
    try:
        src_y = env.unwrapped.config["observation"]["features_range"]["y"]
    except KeyError:
        src_y = [-8, 8]
    try:
        src_vx = env.unwrapped.config["observation"]["features_range"]["vx"]
    except KeyError:
        src_vx = [0, 30]
    try:
        src_vy = env.unwrapped.config["observation"]["features_range"]["vy"]
    except KeyError:
        src_vy = [-5, 5]
    return src_dx, src_y, src_vx, src_vy
