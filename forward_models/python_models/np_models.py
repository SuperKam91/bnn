#########commercial modules
import numpy as np

def relu(x):
	return np.maximum(x, 0.)

def softmax(logits):
	"""
	n.b. logits here aren't log(output) like they are in some definitions.
	using logs and logsumexp trick may be necessary to avoid under/flow
	"""
	return np.exp(logits) / np.sum(np.exp(logits), axis = 1, keepdims = True)

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

def slp(a0, weights):
	"""
	slp
	"""
	z1 = np.matmul(a0, weights[0]) + weights[1]
	return z1

def uap_mlp_ResNet_block(a0, weights):
	a1 = relu(np.matmul(a0, weights[0]) + weights[1])
	return a0 + np.matmul(a1, weights[2])

def coursera_mlp_ResNet_block(a0, weights):
	a1 = relu(np.matmul(a0, weights[0]) + weights[1])
	z2 = a0 + np.matmul(a1, weights[2]) + weights[3]
	return relu(z2)

def mlp_ResNet_1(a0, weights, ResNet_type = 'uap'):
    """
	copied from tf_graphs.py 
    """
    num_blocks = 1
    if ResNet_type == 'uap':
        ResNet_block = uap_mlp_ResNet_block
        weight_slice = 3
    elif ResNet_type == 'coursera':
        ResNet_block = coursera_mlp_ResNet_block
        weight_slice = 4
    else:
        raise NotImplementedError
    for i in range(num_blocks):
    	a2 = ResNet_block(a0, weights[i * weight_slice: (i + 1) * weight_slice])
    	a0 = a2
    return a2

def mlp_ResNet_2(a0, weights, ResNet_type = 'coursera'):
    num_blocks = 2
    if ResNet_type == 'uap':
        ResNet_block = uap_mlp_ResNet_block
        weight_slice = 3
    elif ResNet_type == 'coursera':
        ResNet_block = coursera_mlp_ResNet_block
        weight_slice = 4
    else:
        raise NotImplementedError
    for i in range(num_blocks):
    	a2 = ResNet_block(a0, weights[i * weight_slice: (i + 1) * weight_slice])
    	a0 = a2
    return a2