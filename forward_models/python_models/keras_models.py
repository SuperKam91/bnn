#########commercial modules
import tensorflow as tf
mnist = tf.keras.datasets.mnist
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Regularisers = tf.keras.regularizers

#for information on custom initialisers (for the case of non-trainable parameters), see
#https://keras.io/initializers/

#note to modify model.trainable_weights, have to modify modify model.layers[i].trainable_weights
#BUT NOTE, APPARENTLY when training a model in keras this doesn't affect which parameters are trained
#I should test this at some point.
#for polychord this won't have an effect and the above method is valid

#models and trainable_arr's should be created in tandem

def mlp_test(num_inputs, num_outputs, layer_sizes):
    """
    two parameter mlp test nn
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

def slp_sm(num_inputs, num_outputs, layer_sizes):
	a0 = Input(shape = (num_inputs,))
	prediction = Dense(num_outputs, activation='softmax')(a0)
	return Model(inputs = a0, outputs = prediction)

def slp_sm_r(num_inputs, num_outputs, layer_sizes, reg_coeffs):
	a0 = Input(shape = (num_inputs,))
	prediction = Dense(num_outputs, activation='softmax', kernel_regularizer = Regularisers.l2(reg_coeffs[0]), bias_regularizer = Regularisers.l2(reg_coeffs[1]))(a0)
	return Model(inputs = a0, outputs = prediction)

def mlp_1_sm(num_inputs, num_outputs, layer_sizes):
    """
    simple mlp nn with tanh and softmax activations
    """
    a0 = Input(shape = (num_inputs,))
    a1 = Dense(layer_sizes[0], activation = 'tanh')(a0)
    prediction = Dense(num_outputs, activation='softmax')(a1)
    return Model(inputs = a0, outputs = prediction)

def mlp_1_sm_r(num_inputs, num_outputs, layer_sizes, reg_coeffs):
    """
    in keras, regularisation has to be added layer by layer,
    rather than to cost function
    """
    a0 = Input(shape = (num_inputs,))
    a1 = Dense(layer_sizes[0], activation = 'tanh', kernel_regularizer = Regularisers.l2(reg_coeffs[0]), bias_regularizer = Regularisers.l2(reg_coeffs[1]))(a0)
    prediction = Dense(num_outputs, activation='softmax', kernel_regularizer = Regularisers.l2(reg_coeffs[2]), bias_regularizer = Regularisers.l2(reg_coeffs[3]))(a1)
    return Model(inputs = a0, outputs = prediction)