#########commercial modules
import numpy as np
import tensorflow as tf

#in-house modules
import keras_models as kms

class keras_model():
    """
    class which includes keras model, intended for forward propagation of nn only.
    will eventually contain method that calculates likelihood associated with nn, input and output data.
    note that it assumes model has already been compiled (keras.model.compile(...)) so that loss function is defined.
    steps:
    1) create instance of keras.Model, compile it
    2) create instance of keras_model
    3) setup likelihood function by calling get_LL_const
    4) pass class to polychord function, along with additional arguments (NEED TO BE CONFIRMED)
    --additional notes--
    technically, self.weights isn't really needed at all, just there for testing purposes atm.
    same goes for self.oned_weights. 
    initially thought of keeping nn initialised weights for first iteration, 
    but this would only give values for one livepoint, so should use pc initialisation to initialise weights.
    num_weights also probably not necessary.
    """
    def __init__(self, k_model, x_tr, y_tr, batch_size):
        """
        assign model to class, calculate shape of weights, and arrays containing them (latter possibly redundant)
        """
        self.weight_shapes = []
        self.weights = [] #delete after testing
        self.num_weights = 0 #delete after testing
        self.oned_weights = np.array([]) #possibly delete after testing
        self.model = k_model
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.m = x_tr.shape[0]     
        self.num_outputs = np.prod(np.shape(self.model.layers[-1].output.shape)) #np.prod is in case output isn't vector             
        self.batch_size = batch_size
        self.num_complete_batches = int(np.floor(float(self.m)/self.batch_size))
        self.num_batches = int(np.ceil(float(self.m)/self.batch_size))
        self.get_weight_info()
        
    def calc_gauss_LL(self, x, y, LL_var = 1.):
        """
        WARNING: batch size given here should be same as one given in get_LL_const, or 
        normalisation constant won't be correct.
        as above, only supports scalar variance.
        """
        return - self.LL_dim / (2. * LL_var) * self.model.evaluate(x, y) + self.LL_const 

    def calc_cross_ent_LL(self, x, y):
    	"""
		n.b. for cat cross entr model.evaluate calculates cross entropy then takes average over m.
		uses from_logits=False i.e. does NOT compute softmax for each m, but instead scales each output to
		y_i -> y_i / sum_j y_j. thus it is ADVISABLE to have an explicit softmax layer in your Model
		n.b. requires true y values to be categorical (one-hot) vectors
		including variance in this llikelihood doesn't make sense?
    	"""
    	return - self.m * self.model.evaluate(x, y, batch_size = self.batch_size)
        
    def setup_LL(self, LL_var = 1.):
        """
        calculates LL constant, and sets correct LL function, creates generator object for batches.
        currently only supports single (scalar) variance across all records and outputs,
        since we use model.evaluate() to calculate cost with predefined loss functions (e.g. mse, crossentropy) 
        which evaluates sums across examples/outputs
        (so variance can't be included in each summation).
        if we instead calculate likelihood from final layer output, will probably use scipy.stats to do so
        and this function will become redundant. 
        if instead we define own loss function which allows for different variances,
        comment out lines below (and variance should be LL_dim x LL_dim array) and uncomment ones further down
        """
        if self.m <= self.batch_size:
            self.batch_generator = None
        else:
            self.batch_generator = self.create_batch_generator()
        self.LL_dim = self.batch_size * self.num_outputs
        if self.model.loss == 'mse':
            #temporary
            self.LL_const = -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(LL_var))
            self.LL = self.calc_gauss_LL
            #longer term solution (see comments above)
            #self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
        elif self.model.loss == 'categorical_crossentropy':
        	self.LL_const = 0.
        	self.LL = self.calc_cross_ent_LL
        else:
            raise NotImplementedError
        
    def get_weight_info(self):
        """
        weight arrays may be redundant (see above)
        """
        for layer_weight in self.model.get_weights():
            layer_shape = layer_weight.shape
            self.weight_shapes.append(layer_shape)
            self.weights.append(layer_weight) #delete after testing
            self.oned_weights = np.concatenate((self.oned_weights, layer_weight.reshape(-1))) #possibly delete after testing
            self.num_weights += np.prod(layer_shape) #delete after testing
        
    def get_weight_shapes(self):
        return self.weight_shapes
    
    def get_weights(self): #delete after testing
        return self.weights
    
    def get_oned_weights(self): #possibly delete after testing
        return self.oned_weights
    
    def get_oned_weights2(self): #possibly delete after testing
        """
        used in case where we want array of nn initial weights to pass to pc, but after that
        it can be deleted.
        WARNING: deletes self.oned_weights
        """
        temp = self.oned_weights
        del self.oned_weights
        return temp
    
    def get_num_weights(self): #delete after testing
        return self.num_weights
    
    def get_model(self):
        """
        return keras model instance
        """
        return self.model
    
    def get_model_summary(self):
        """
        return keras model summary
        """
        return self.model.summary()
    
    def get_model_weights(self):
        """
        returns list of weight arrays (one element for each layer's set of weight/bias)
        """
        return self.model.get_weights()
    
    def set_weights(self, weights): #delete after testing
        self.weights = weights
        
    def set_oned_weights(self, oned_weights): #possibly delete after testing
        self.oned_weights = oned_weights
        
    def set_model_weights(self, weights):
        self.model.set_weights(weights)
        
    def set_k_weights(self, new_oned_weights):
        """
        set weights of keras.Model using 1d array of weights.
        beside this, updates weight array attributes (which may be deleted after testing).
        """
        self.set_oned_weights(new_oned_weights) #possibly delete after testing
        new_weights = []
        start_index = 0
        for weight_shape in self.get_weight_shapes():
            weight_size = np.prod(weight_shape)
            new_weights.append(new_oned_weights[start_index:start_index + weight_size].reshape(weight_shape))
            start_index += weight_size
        self.set_weights(new_weights) #delete after testing
        self.set_model_weights(new_weights)
        
    def __call__(self, oned_weights):
        """
        sets keras.Model weights, gets new batch of training data (or full batch), 
        evaluates log likelihood function and returns its value.
        to be passed to polychord as loglikelihood function
        """
        self.set_k_weights(oned_weights)
        x_batch, y_batch = self.get_batch()
        LL = self.LL(x_batch, y_batch)
        return LL
        
    def get_batch(self):
        """
        returns either entire training data, or uses batch generator object to generate
        new batch.
        """
        if self.m <= self.batch_size:
            return self.x_tr, self.y_tr
        else:
            return self.batch_generator.next()
            
    def create_batch_generator(self):
        """
        creates a batch generator object which yields a batch (subset) of the training data
        from a list of batches created by create_batches(). in the case that the end of the list
        is reached, calls create_batches to create a new list of batches and returns first in list.
        """
        i = 0
        batches = self.create_batches()
        while True:
            if i < self.num_batches:
                pass #don't need to create new random shuffle of training data
            else:
                batches = self.create_batches()
                i = 0
            yield batches[i]
            i += 1
    
    def create_batches(self):
        """
        create batches of size self.batch_size from self.m training examples
        for training.
        In case of large training sets, batches may want to overwrite self.x_tr/self.y_tr
        (if not both have to be saved to memory as batches is saved in generator object).
        NOTE: in case of batch_size not being factor of m, last batch in list is of size
        < batch_size, so likelihood calculation is INCORRECT (normalisation constant).
        even if used correct normalisation constant, wouldn't be consistent with other calculations.
        Thus for Bayesian problems we should probably just discard these 
        extra training examples and ensure batch_size / m is an integer.
        """
        batches = []
        # Step 1: Shuffle x, y
        permutation = np.random.permutation(self.m)
        shuffled_x = self.x_tr[permutation]
        shuffled_y = self.y_tr[permutation]
        # Step 2: Partition (shuffled_x, shuffled_y). Minus the end case.
        # number of batches of size self.batch_size in your partitionning
        for i in range(self.num_complete_batches):
            batch_x = shuffled_x[self.batch_size * i: self.batch_size * (i + 1)]
            batch_y = shuffled_y[self.batch_size * i: self.batch_size * (i + 1)]
            batch = (batch_x, batch_y)
            batches.append(batch)
    
        # Handling the end case (last batch < self.batch_size)
        if self.num_complete_batches != self.num_batches:
            batch_x = shuffled_x[self.num_complete_batches * self.batch_size:]
            batch_y = shuffled_y[self.num_complete_batches * self.batch_size:]
            batch = (batch_x, batch_y)
            batches.append(batch)
        return batches

def main():

	num_inputs = 2
	num_outputs = 2
	m = 3
	batch_size = 3

	model = kms.slp_model(num_inputs, num_outputs)
	model.compile(loss='mse', optimizer='rmsprop')
	np.random.seed(1337)
	x_tr = np.random.random((m, num_inputs))
	y_tr = np.array([1,0,0,1,1,0]).reshape(3,2)
	km = keras_model(model, x_tr, y_tr, batch_size) 
	km.setup_LL()
	km(np.arange(27))

if __name__ == '__main__':
	main()


    
        

