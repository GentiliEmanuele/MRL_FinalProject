import numpy as np


class WeightsHandler:
    def __init__(self, states_len, actions_len):
        self.shape = (states_len, actions_len)

    def generate_weights(self):
        weights = np.random.rand(*self.shape)
        return weights

    def save_weights(self, weights, filename):
        if weights is not None:
            np.save(filename, weights)
        else:
            print("No weights to save. Generate weights first.")

    def load_weights(self, filename):
        try:
            weights = np.load(filename)
            return weights
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading weights: {e}")
            return None