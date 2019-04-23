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
	"""
	basic regression architecture using single layer perceptron
	(softmax activation)
	"""
	a0 = Input(shape = (num_inputs,))
	prediction = Dense(num_outputs, activation='softmax')(a0)
	return Model(inputs = a0, outputs = prediction)

def slp_sm_r(num_inputs, num_outputs, layer_sizes, reg_coeffs):
	"""
	as above but regularised
	"""
	a0 = Input(shape = (num_inputs,))
	prediction = Dense(num_outputs, activation='softmax', kernel_regularizer = Regularisers.l2(reg_coeffs[0]), bias_regularizer = Regularisers.l2(reg_coeffs[1]))(a0)
	return Model(inputs = a0, outputs = prediction)

def slp(num_inputs, num_outputs, layer_sizes):
	"""
	basic regression architecture using single layer perceptron
	(linear activation)
	"""
	a0 = Input(shape = (num_inputs,))
	prediction = Dense(num_outputs, activation='linear')(a0)
	return Model(inputs = a0, outputs = prediction)

def slp_r(num_inputs, num_outputs, layer_sizes, reg_coeffs):
	"""
	as above but regularised
	"""
	a0 = Input(shape = (num_inputs,))
	prediction = Dense(num_outputs, activation='linear', kernel_regularizer = Regularisers.l2(reg_coeffs[0]), bias_regularizer = Regularisers.l2(reg_coeffs[1]))(a0)
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

def uap_mlp_ResNet_block(num_inputs, a0):
    """
    in this architecture each block consists of two layers, the first
    being one neuron wide, the second being num_inputs wide.
    the first layer contains a bias and a ReLu activation,
    the second does not.
    """
    a1 = Dense(1, activation = 'relu')(a0)
    a2_part = Dense(num_inputs, activation = 'linear', use_bias = False)(a1)
    return tf.keras.layers.Add()([a0, a2_part])

def coursera_mlp_ResNet_block(num_inputs, a0):
    """
    same as uap_mlp_ResNet_block() but second layer also contains
    bias and ReLu activation. 
    """
    a1 = Dense(1, activation = 'relu')(a0)
    z2_part = Dense(num_inputs, activation = 'linear')(a1)
    z2 = tf.keras.layers.Add()([a0, z2_part])
    return tf.keras.layers.Activation('relu')(z2)

def same_mlp_ResNet_block(num_inputs, a0):
    """
    in this ResNet architecture, intermediate (a1) layer
    has same dimensions as a0 (input)/a2 (output). 
    Uses coursera convention for output layer as above
    (output includes bias and activation)
    """
    a1 = Dense(num_inputs, activation = 'relu')(a0)
    z2_part = Dense(num_inputs, activation = 'linear')(a1)
    z2 = tf.keras.layers.Add()([a0, z2_part])
    return tf.keras.layers.Activation('relu')(z2)

def mlp_ResNet_1(num_inputs, num_outputs, layer_sizes, ResNet_type = 'uap'):
    """
    layer_sizes not actually needed, just included in signature
    for consistency.
    could build a function to handle arbitrary num_blocks, but cba init.
    could just pass num_blocks as argument, as model is created in main,
    but don't to be consistent with np, tf and cpp implementations.
    one block ResNet
    """
    num_blocks = 1
    if ResNet_type == 'uap':
        ResNet_block = uap_mlp_ResNet_block
    elif ResNet_type == 'coursera':
        ResNet_block = coursera_mlp_ResNet_block
    elif ResNet_type == 'same':
        ResNet_block = same_mlp_ResNet_block
    else:
        raise NotImplementedError
    a0 = Input(shape = (num_inputs,))
    inputs = a0
    for _ in range(num_blocks):
        a2 = ResNet_block(num_inputs, a0)
        a0 = a2
    return Model(inputs = inputs, outputs = a2)

def mlp_ResNet_2(num_inputs, num_outputs, layer_sizes, ResNet_type = 'uap'):
    """
    two block ResNet
    """
    num_blocks = 2
    if ResNet_type == 'uap':
        ResNet_block = uap_mlp_ResNet_block
    elif ResNet_type == 'coursera':
        ResNet_block = coursera_mlp_ResNet_block
    elif ResNet_type == 'same':
        ResNet_block = same_mlp_ResNet_block
    else:
        raise NotImplementedError
    a0 = Input(shape = (num_inputs,))
    inputs = a0
    for _ in range(num_blocks):
        a2 = ResNet_block(num_inputs, a0)
        a0 = a2
    return Model(inputs = inputs, outputs = a2)