#########commercial modules
import numpy as np
import tensorflow as tf

#in-house modules
import keras_models as kms
import tools
import PyPolyChord
import PyPolyChord.settings
import inverse_priors
import polychord_tools
import output
import input_tools
import prior_tests
import forward_tests

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
        # self.weights = [] #delete after testing
        # self.num_weights = 0 #delete after testing
        # self.oned_weights = np.array([]) #possibly delete after testing
        self.model = k_model
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.m = x_tr.shape[0]     
        self.num_outputs = np.prod(np.shape(self.model.layers[-1].output.shape[1:])) #np.prod is in case output isn't vector             
        self.batch_size = batch_size
        #possibly could use boolean for whether remainder batch is needed, 
        #but can't be bothered as would require modifying batch functions
        self.num_complete_batches = int(np.floor(float(self.m)/self.batch_size))
        self.num_batches = int(np.ceil(float(self.m)/self.batch_size))
        self.get_weight_info()
        self.LL_var = 1. #take this as an argument in the future probably, either in init or ll_setup
        
    def calc_gauss_LL(self, x, y):
        """
        note batch_size passed to .evaluate here is irrelevant to the LL value calculated for bayesian inference,
        as the total batch (x, y) is used regardless of its value to calculate ll (by multiplying together
        the values calculated from each batch of size batch_size [plus remainder] passed to .evaluate), 
        it merely dictactes in what chunks the loss is calculated i.e. .evaluate(batch_size). 
        this process is most efficient when the batch_size is large (= m in full batch case, = batch_size in batch case).
        if the batch_size passed to .evaluate > size(x/y), calculates loss in one full chunk.
        to get LL of mini batches (< m), use mini batch functions below to split m.
        NOTE when using keras pipeline (and using .fit) the above statement doesn't apply as each batch
        is back propagated and the weights are updated before moving onto the next batch. thus when using keras
        pipeline the batch functions below can be ignored (if const var), and the batch_size specified here.
        as above, only supports scalar variance.
        the steps argument to .evaluate is another way to specify batch_size (batch_size = m / steps), but still
        evaluates full x,y
        """
        LL = - self.LL_dim / (2. * self.LL_var) * self.model.evaluate(x, y, batch_size = self.batch_size, verbose = 0) + self.LL_const  
        return LL

    def calc_cross_ent_LL(self, x, y):
    	"""
		n.b. for cat cross entr model.evaluate calculates cross entropy then takes average over batch_size.
        CHECK AVERAGE IS OVER BATCH_SIZE AND NOT LL_DIM
		uses from_logits=False i.e. does NOT compute softmax for each m, but instead scales each output to
		y_i -> y_i / sum_j y_j. thus it is ADVISABLE to have an explicit softmax layer in your Model
		n.b. requires true y values to be categorical (one-hot) vectors
		including variance in this llikelihood doesn't make sense?
    	"""
    	return - self.batch_size * self.model.evaluate(x, y, batch_size = self.batch_size, verbose = 0)
        
    def setup_LL(self, loss):
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
        self.model.compile(loss=loss, optimizer='rmsprop') #optimiser is irrelevant for this class
        if self.m <= self.batch_size:
            self.batch_generator = None
        else:
            self.batch_generator = self.create_batch_generator()
        self.LL_dim = self.batch_size * self.num_outputs
        if self.model.loss == 'mse':
            #temporary
            self.LL_const = -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var))
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
        updated to just get weight_shapes of trainable weight matrices/ bias vectors.
        for info on how to make specific parameters non-trainable see:
        https://stackoverflow.com/questions/42741402/keras-trainable-weight-issue
        """
        trainable_weights = tf.keras.backend.get_session().run(self.model.trainable_weights)
        # for layer_weight in self.model.get_weights():
        for layer_weight in trainable_weights:
            layer_shape = layer_weight.shape
            self.weight_shapes.append(layer_shape)
            # self.weights.append(layer_weight) #delete after testing
            # self.oned_weights = np.concatenate((self.oned_weights, layer_weight.reshape(-1))) #possibly delete after testing
            # self.num_weights += np.prod(layer_shape) #delete after testing

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
        by calling keras model method
        """
        return self.model.summary()
    
    def get_model_weights(self):
        """
        returns list of weight arrays (one element for each layer's set of weight/bias)
        by calling keras model method
        """
        return self.model.get_weights()
    
    def set_weights(self, weights): #delete after testing
        self.weights = weights
        
    def set_oned_weights(self, oned_weights): #possibly delete after testing
        self.oned_weights = oned_weights
        
    def set_model_weights(self, weights):
        """
        calls keras method to set weights of keras model
        """
        self.model.set_weights(weights)
        
    def set_k_weights(self, new_oned_weights):
        """
        set weights of keras.Model using 1d array of weights.
        beside this, updates weight array attributes (which may be deleted after testing).
        note, as desirable, ignores non-trainable weights
        n.b. weight matrix shapes are such that weight matrix multiplies previous activiation to the right.
        n.b. numpy arrays are by default stored in rowmajor order
        """
        # self.set_oned_weights(new_oned_weights) #possibly delete after testing
        new_weights = []
        start_index = 0
        for weight_shape in self.get_weight_shapes():
            weight_size = np.prod(weight_shape)
            new_weights.append(new_oned_weights[start_index:start_index + weight_size].reshape(weight_shape))
            start_index += weight_size
        # self.set_weights(new_weights) #delete after testing
        self.set_model_weights(new_weights)
        
    def __call__(self, oned_weights):
        """
        sets keras.Model weights, gets new batch of training data (or full batch), 
        evaluates log likelihood function and returns its value.
        to be passed to polychord as loglikelihood function
        """
        self.set_k_weights(oned_weights)
        x_batch, y_batch = self.get_batch()
        #if non-constant variance ever necessary,
        #self.LL_var and LL_const will have to be updated here, 
        #same goes for tf and np forward classes 
        LL = self.LL(x_batch, y_batch)
        return LL
        
    def get_batch(self):
        """
        returns either entire training data, or uses batch generator object to generate
        new batch.
        even though keras has built-in batch facilities, for bayesian inference these aren't useful.
        see calculating llhood methods for more details.
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
        if different methods adopted in create_batches, this function will also need to be modified
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
        quickest way to make this more efficient is get rid of shuffled_x/y and just use permutation
        to create list with elements being shuffled x,y, so only one extra copy of x,y required.
        better still, could re-implement this so only one copy of training data is required and 
        just split permutation vector into mini batches (a list of parts of the permutation vector)
        and use elements of list to slice original dataset and return this in yield 
        (though each batch will be a copy of subset of x,y), then only list of permutation 
        (and maybe original permutation array) and single batch needs to be stored each iteration
        (fancying indexing of array creates copy, not view).
        or could overwrite self.tr data with permuted data, then probably don't need permutation
        vector either, and batch data can be a view of full data rather than a copy of subset of x,y
        if batch_size small, don't mind making copy for batch, copy of permutation / shuffled data
        doesn't need to be made so often (similarly overwriting x,y not needed often), so perhaps in-place
        or copy methods would work fine
        if batch_size large, copy of batch could be expensive, and copy of permutation / shuffled data
        needs to be made often (similarly overwriting x,y needed often), so in-place methods probably better
        if width of x,y large, permutation vector much less costly than making copies of data or shuffling
        in-place. so overall method depends on batch_size and width
        NOTE: in case of batch_size not being factor of m, last batch in list is of size
        < batch_size, so likelihood calculation is INCORRECT (normalisation constant).
        even if used correct normalisation constant, wouldn't be consistent with other calculations.
        Thus for Bayesian problems we should probably just discard these 
        extra training examples and ensure batch_size / m is an integer.
        in case of each training data having its own variance, would also need list of these here.
        """
        batches = []
        # Step 1: Shuffle x, y
        permutation = np.random.permutation(self.m)
        #these create copies (not views)
        shuffled_x = self.x_tr[permutation]
        shuffled_y = self.y_tr[permutation]
        # Step 2: Partition (shuffled_x, shuffled_y). Minus the end case.
        # number of batches of size self.batch_size in your partitionning
        #note this essentially makes a second copy
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

def main(run_string):
    ###### load training data
    data = 'simple_tanh'
    data_dir = '../../data/'
    data_prefix = data_dir + data
    x_tr, y_tr = input_tools.get_x_y_tr_data(data_prefix)
    batch_size = x_tr.shape[0]
    ###### get weight information
    a1_size = 2
    layer_sizes = [a1_size]
    m_trainable_arr = [True, False]
    b_trainable_arr = [False, False]
    num_inputs = tools.get_num_inputs(x_tr)
    num_outputs = tools.get_num_outputs(y_tr)
    num_weights = tools.calc_num_weights3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	###### check shapes of training data
	x_tr, y_tr = tools.reshape_x_y_twod(x_tr, y_tr)
	###### setup keras model
	model = kms.slp_model(num_inputs, num_outputs, layer_sizes)
    km = keras_model(model, x_tr, y_tr, batch_size)
    loss = 'mse' 
    km.setup_LL(loss)
    ###### test llhood output
    if "forward_test_linear" in run_string:
    	forward_tests.forward_test_linear([km], num_weights)
    ###### setup prior
    prior_types = [7]
	prior_hyperparams = [[-2., 2.]]
	weight_shapes = get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
	dependence_lengths = get_degen_dependence_lengths(weight_shapes)
	param_prior_types = [0]
	prior = inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, num_weights)
	###### test prior output from nn setup
	if "nn_prior_test" in run_string:
		prior_tests.nn_prior_test(prior)
    ###### setup polychord
    nDerived = 0
    settings = PyPolyChord.settings.PolyChordSettings(num_weights, nDerived)
    settings.base_dir = './keras_chains/'
    settings.file_root = data
    settings.nlive = 200
    ###### run polychord
    if "polychord1" in run_string:
    	PyPolyChord.run_polychord(km, num_weights, nDerived, settings, prior, polychord_tools.dumper)

if __name__ == '__main__':
	run_string = ''
	main(run_string)


    
        

