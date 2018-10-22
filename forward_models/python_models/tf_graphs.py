#########commercial modules
import tensorflow as tf

#further to comments in keras_models.py, it is crucial to use the correct tf_model class
#depending on whether layers / matrices / vectors are frozen 

def slp_graph(a0, weights):
    """
    tf graph builder for single layer perceptron classification nn.
    uses convention that weight matrices act on previous activation to the right.
    n.b. by default tf and np add 1-d arrays and row vectors to matrices row-wise,
    but column vectors column-wise
    """
    a1 = tf.tanh(tf.matmul(a0, weights[0]))
    prediction = tf.reduce_sum(a1, axis = 1, keepdims = True)
    return prediction
