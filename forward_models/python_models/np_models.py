#########commercial modules
import numpy as np

def relu(x):
	return np.maximum(x, 0)

def slp_nn(a0, weights):
    """
    based on tf version in tf_graphs.py
    """
    a1 = np.tanh(np.matmul(a0, weights[0]))
    prediction = np.sum(a1, axis = 1, keepdims = True)
    return prediction
