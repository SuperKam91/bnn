#########commercial modules
import tensorflow as tf
mnist = tf.keras.datasets.mnist
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense

#for information on custom initialisers (for the case of non-trainable parameters), see
#https://keras.io/initializers/

#note to modify model.trainable_weights, have to modify modify model.layers[i].trainable_weights
#BUT NOTE, APPARENTLY when training a model in keras this doesn't affect which parameters are trained
#I should test this at some point.
#for polychord this won't have an effect and the above method is valid

#models and trainable_arr's should be created in tandem

def slp_model(num_inputs, num_outputs, layer_sizes):
    """
    keras model builder (using model api) for single layer perceptron classification nn.
    uses convention that weight matrices act on previous activation to the right.
    """
    a0 = Input(shape = (num_inputs,))
    a1 = Dense(layer_sizes[0], activation = 'tanh', bias_initializer = tf.keras.initializers.Zeros)(a0)
    prediction = Dense(num_outputs, activation = 'linear', trainable = False, kernel_initializer = tf.keras.initializers.Ones, bias_initializer = tf.keras.initializers.Zeros) (a1)
    model = Model(inputs = a0, outputs = prediction)
    #add bias of hidden layer to non_trainable_weights
    model.layers[1].non_trainable_weights.append(model.layers[1].trainable_weights[1])
    #remove bias of hidden layer from trainable_weights
    del model.layers[1].trainable_weights[1]
    return model