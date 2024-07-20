import pickle


def serilizeIHT(iht, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(iht, file)


def deserializeIHT(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)
