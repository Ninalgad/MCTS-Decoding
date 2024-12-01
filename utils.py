import numpy as np
from scipy.special import softmax


def softmax_sample(items, distribution, temperature: float):
    distribution = np.array(distribution, "float32")
    probs = softmax(distribution / temperature)
    idx = np.random.choice(len(items), p=probs)
    return idx, items[idx]

