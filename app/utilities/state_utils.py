import numpy as np


# wrapper for env.configure(config)
def custom_configure(env, config):
    # get the value of ["observation"]["normalize"], or the default one
    try:
        normalize = config["observation"]["normalize"]
    except KeyError:
        normalize = True
    # get the value of ["observation"]["absolute"], or the default one
    try:
        absolute = config["observation"]["absolute"]
    except KeyError:
        absolute = False

    t_config = config.copy()
    # if observation are absolute, then the normalization is broken,
    # so we set "normalize" = False, and then we normalize in our way
    if absolute:
        t_config["observation"]["normalize"] = False
        t_config["observation"]["custom_normalize"] = normalize
    env.configure(t_config)


# wrapper for state, info = env.reset(seed=cu.get_seed())
def custom_reset(env, custom_seed):
    state, info = env.reset(seed=custom_seed)
    # apply our transformation for the state
    t_state, nt_state = transform_state(env, state)
    return t_state, info, nt_state


# wrapper for state, reward, done, truncated, info = env.step(action)
def custom_step(env, action):
    state, reward, done, truncated, info = env.step(action)
    # apply our transformation for the state
    t_state, nt_state = transform_state(env, state)
    return t_state, reward, done, truncated, info, nt_state


def transform_state(env, state):
    # get the value of ["observation"]["normalize"], or the default one
    try:
        normalize = env.unwrapped.config["observation"]["normalize"]
    except KeyError:
        normalize = True
    # get the value of ["observation"]["custom_normalize"], or the default one
    try:
        custom_normalize = env.unwrapped.config["observation"]["custom_normalize"]
    except KeyError:
        custom_normalize = True
    # get the value of ["observation"]["normalize_scale"], or the default one
    try:
        normalize_scale = env.unwrapped.config["observation"]["normalize_scale"]
    except KeyError:
        normalize_scale = [0, 0]
    # get the value of ["observation"]["absolute"], or the default one
    try:
        is_absolute = env.unwrapped.config["observation"]["absolute"]
    except KeyError:
        is_absolute = False

    # is observation are absolute, then we want to use the distance along x,
    # instead of the x values
    if is_absolute:
        # subtract to all rows the x value of the agent
        agent_x = state[0][0]
        state[:, 0] -= agent_x
        # if absolute and normalize, then normalize in our way
        if custom_normalize:
            state = normalize_state(env, state)

    # if normalize AND normalize_scale, then we apply a scaling,
    # this is done to make the IHT consider near states more far apart,
    # increasing the difference in tilings
    if (normalize or custom_normalize) and normalize_scale != [0, 0]:
        if is_absolute:
            state[:, :] = np.interp(state[:, :], [0, 1], normalize_scale)
        else:
            state[:, :] = np.interp(state[:, :], [-1, 1], normalize_scale)

    nt_state = state

    # we return the flattened state as wanted by tiles(...),
    # as long as the state in the original shape for printing
    return state.flatten()[1:].tolist(), nt_state


# custom normalization for absolute state
def normalize_state(env, state):
    src_dx, src_y, src_vx, src_vy = get_bounds(env)
    state[:, 0] = np.interp(state[:, 0], src_dx, [0, 1])
    state[:, 1] = np.interp(state[:, 1], src_y,  [0, 1])
    state[:, 2] = np.interp(state[:, 2], src_vx, [0, 1])
    state[:, 3] = np.interp(state[:, 3], src_vy, [0, 1])
    return state


# read bounds from configuration, or use the default value
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
