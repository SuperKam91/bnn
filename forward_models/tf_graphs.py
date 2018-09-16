#########commercial modules
import tensorflow as tf

def slp_graph(a0, weights):
    """
    tf graph builder for single layer perceptron classification nn
    """
    a1 = tf.nn.relu(tf.matmul(a0, weights[0]) + weights[1])
    prediction = tf.matmul(a1, weights[2]) + weights[3]
    return prediction