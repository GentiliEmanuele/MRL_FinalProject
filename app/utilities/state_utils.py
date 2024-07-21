def transform_state(state):
    agent_x = state[0][0]
    state[:, 0] -= agent_x
    stat = normalize_state(state)
    return state.flatten()[1:].tolist()

def normalize_state(state):
     state[:, 0] /= 200