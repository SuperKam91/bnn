#########commercial modules
import tensorflow as tf
mnist = tf.keras.datasets.mnist
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense

def slp_model(num_inputs, num_outputs, layer_sizes):
    """
    keras model builder (using model api) for single layer perceptron classification nn.
    uses convention that weight matrices act on previous activation to the right.
    """
    a0 = Input(shape = (num_inputs,))
    a1 = Dense(layer_sizes[0], activation = 'relu')(a0)
    prediction = Dense(num_outputs, activation='linear')(a1)
    return Model(inputs = a0, outputs = prediction)