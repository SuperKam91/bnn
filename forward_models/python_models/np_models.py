#########commercial modules
import numpy as np

def slp_nn(a0, weights):
    """
    based on tf version in tf_graphs.py
    """
    a1 = np.maximum(np.matmul(a0, weights[0]) + weights[1], 0.)
    prediction = np.matmul(a1, weights[2]) + weights[3]
    return prediction
