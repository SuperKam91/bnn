#########commercial modules
import tensorflow as tf

#further to comments in keras_models.py, it is crucial to use the correct tf_model class
#depending on whether layers / matrices / vectors are frozen 

def mlp_test(a0, weights):
    """
    two parameter mlp test nn
    n.b. by default tf and np add 1-d arrays and row vectors to matrices row-wise,
    but column vectors column-wise
    """
    a1 = tf.tanh(tf.matmul(a0, weights[0]))
    prediction = tf.reduce_sum(a1, axis = 1, keepdims = True)
    return prediction

def slp_sm(a0, weights):
	"""
	softmax slp
	"""
	z1 = tf.matmul(a0, weights[0]) + weights[1]
	# might have to take log of z1 to get appropriate logits for tf cross entropy 
	return tf.nn.softmax(z1)

def slp(a0, weights):
	z1 = tf.matmul(a0, weights[0]) + weights[1]
	return z1

def uap_mlp_ResNet_block(a0, weights):
	a1 = tf.nn.relu(tf.matmul(a0, weights[0]) + weights[1])
	return a0 + tf.matmul(a1, weights[2])

def coursera_mlp_ResNet_block(a0, weights):
	a1 = tf.nn.relu(tf.matmul(a0, weights[0]) + weights[1])
	z2 = a0 + tf.matmul(a1, weights[2]) + weights[3]
	return tf.nn.relu(z2)

def mlp_ResNet_1(a0, weights, ResNet_type = 'uap'):
    """
	could create a function for arbitrary number of blocks,
	but graph isn't created in main, it is created in tf_forward class,
	so can't specify from main without reconfiguring which is undesirable.
	could calculate num_blocks from size of weights, but requires assumptions on rest
	of network (i.e. that there are no frozen weight sets in the other layers), but cba init
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