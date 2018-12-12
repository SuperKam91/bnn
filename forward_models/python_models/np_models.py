#########commercial modules
import numpy as np

def relu(x):
	return np.maximum(x, 0.)

def softmax(logits):
	"""
	n.b. logits here aren't log(output) like they are in some definitions.
	using logs and logsumexp trick may be necessary to avoid under/flow
	"""
	return np.exp(logits) / np.sum(np.exp(logits), axis = 1, keepdims=True)

def mlp_test(a0, weights):
	"""
	based on tf version in tf_graphs.py
	"""
	a1 = np.tanh(np.matmul(a0, weights[0]))
	prediction = np.sum(a1, axis = 1, keepdims = True)
	return prediction

def slp_sm(a0, weights):
	"""
	softmax slp
	"""
	z1 = np.matmul(a0, weights[0]) + weights[1]
	return softmax(z1)