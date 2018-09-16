#########commercial modules
import tensorflow as tf
mnist = tf.keras.datasets.mnist
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense

def slp_model(num_inputs, num_outputs):
    """
    keras model builder (using model api) for single layer perceptron classification nn
    """
    a0 = Input(shape = (num_inputs,))
    a1 = Dense(5, activation = 'relu')(a0)
    prediction = Dense(num_outputs, activation='linear')(a1)
    return Model(inputs = a0, outputs = prediction)