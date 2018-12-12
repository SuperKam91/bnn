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