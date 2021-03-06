#for some reason if you import scipy.stats before tf, get ImportError on scipy.stats import
import inverse_priors
import inverse_stoc_hyper_priors as isp
import inverse_stoc_var_hyper_priors as isvp

#########commercial modules
import numpy as np
import tensorflow as tf

#in-house modules
import keras_models as kms
import tools
import PyPolyChord
import PyPolyChord.settings
import polychord_tools
import input_tools
import prior_tests
import forward_tests
import output_tools

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
    def __init__(self, k_model, x_tr, y_tr, batch_size, n_stoc_var = 0):
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
        self.num_outputs = np.prod(self.model.layers[-1].output.shape[1:].as_list()) #np.prod is in case output isn't vector
        self.batch_size = batch_size
        #possibly could use boolean for whether remainder batch is needed, 
        #but can't be bothered as would require modifying batch functions
        self.num_complete_batches = int(np.floor(float(self.m)/self.batch_size))
        self.num_batches = int(np.ceil(float(self.m)/self.batch_size))
        self.get_weight_info()
        self.LL_var = 1. #take this as an argument in the future probably, either in init or ll_setup
        self.n_stoc_var = n_stoc_var #currently only works for 0 or 1.

    def squared_error_LL_c(self):
    	"""
		ll const for gauss lhood
    	"""
    	return -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var))

    def categorical_crossentropy_LL_c(self):
    	"""
		ll const for multinomial lhood
    	"""
    	return 0.

    def av_squared_error_LL_c(self):
    	"""
		ll const for av gauss lhood
    	"""
    	return -0.5 * self.LL_dim * (np.log(2. * np.pi) + np.log(self.LL_var) + np.log(self.LL_dim))

    def av_categorical_crossentropy_LL_c(self):
    	"""
		ll const for av multinomial lhood
    	"""
    	return 0.

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
        if self.m <= self.batch_size:
            self.batch_generator = None
        else:
            self.batch_generator = self.create_batch_generator()
        self.LL_dim = self.batch_size * self.num_outputs
        if loss == 'squared_error':
            self.model.compile(loss='mse', optimizer='rmsprop') #optimiser is irrelevant for this class.'unaverages' mse in call to calc_gauss_ll
            #temporary
            self.LL_const_f = self.squared_error_LL_c
            self.LL = self.calc_gauss_LL
            #longer term solution (see comments above)
            #self.LL_const = -0.5 * (LL_dim * np.log(2. * np.pi) + np.log(np.linalg.det(variance)))
        elif loss == 'categorical_crossentropy':
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop') #'unaverages' loss in call to calc_cross_ent_LL
            self.LL_const_f = self.categorical_crossentropy_LL_c
            self.LL = self.calc_cross_ent_LL
        elif loss == 'av_squared_error':
            self.model.compile(loss='mse', optimizer='rmsprop') 
            #temporary
            self.LL_const_f = self.av_squared_error_LL_c
            self.LL = self.calc_av_gauss_LL
        elif loss == 'av_categorical_crossentropy':
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
            self.LL_const_f = self.av_categorical_crossentropy_LL_c
            self.LL = self.calc_av_cross_ent_LL
        else:
            raise NotImplementedError
        self.LL_const = self.LL_const_f() 
        self.stoc_var_setup()

#for LL calculations, may want to consider taking average as this is what MLE/MPE estimates do, meaning they essentially sample
#from a different likelihood (MPE samples from LL with a variance which is a factor of batch_size larger).
#furthermore for MPE, using optimisation techniques which use gradients e.g. gradient descent, derivative is a factor of 
#batch_size smaller, so the stepsize is also effected

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

    def calc_av_gauss_LL(self, x, y):
        """
        adapted from non-average version.
        see np_forward.py implementation for more
        details concerning 'average'.
        """
        LL = - 1 / (2. * self.LL_var) * self.model.evaluate(x, y, batch_size = self.batch_size, verbose = 0) + self.LL_const  
        return LL

    def calc_cross_ent_LL(self, x, y):
    	"""
		n.b. for cat cross entr model.evaluate calculates cross entropy.
		uses from_logits=False i.e. does NOT compute softmax for each m, but instead scales each output to
		y_i -> y_i / sum_j y_j, see: https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/backend.py
		thus it is ADVISABLE to have an explicit softmax layer in your Model.
		n.b. model.evaluate calculates cross entropy for each record in batch_size, then averages.
		n.b. requires true y values to be categorical (one-hot) vectors
		including variance in this llikelihood doesn't make sense?
		n.b. LL = batch_size * categorical_cross_ent
    	"""
    	return - self.batch_size * self.model.evaluate(x, y, batch_size = self.batch_size, verbose = 0)

    def calc_av_cross_ent_LL(self, x, y):
        """
        adapted from non-average version.
        see np_forward.py implementation for more
        details concerning 'average'.
        """
        self.LL_const = -1 * np.log((self.model.predict(x)**(1. / self.batch_size)).prod(axis = 0).sum())
        return - 1. * self.model.evaluate(x, y, batch_size = self.batch_size, verbose = 0) + self.LL_const

    def stoc_var_setup(self):
    	"""
		stochastic vars only currently supports 0 or 1 stochastic variances
    	"""
    	if self.n_stoc_var == 0:
    		self.stoc_var_update = self.no_stoc_var_update
    	elif self.n_stoc_var == 1:
    		self.stoc_var_update = self.one_stoc_var_update
    	else:
    		print "only 0 or 1 stoc variances currently supported"
    		raise NotImplementedError

    def no_stoc_var_update(self, LL_vars):
    	"""
		if no stochastic variances, no updating needs to be done here
    	"""
    	return None

    def one_stoc_var_update(self, LL_vars):
    	"""
		update variance, recalculate llhood normalisation constant
    	"""
    	self.LL_var = LL_vars[0]
    	self.LL_const = self.LL_const_f()
        
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
        self.set_k_weights(oned_weights[self.n_stoc_var:])
        x_batch, y_batch = self.get_batch()
        self.stoc_var_update(oned_weights[:self.n_stoc_var])
        LL = self.LL(x_batch, y_batch)
        return LL

    def test_output(self, oned_weights):
        print "one-d weights:"
        print oned_weights
        self.set_k_weights(oned_weights[self.n_stoc_var:])
        x_batch, y_batch = self.get_batch()
        print "input batch:"
        print x_batch
        print "output batch:"
        print y_batch
        print "nn output:"
        print self.model.predict(x_batch)
        self.stoc_var_update(oned_weights[:self.n_stoc_var])
        print "LL var and const"
        print self.LL_var, self.LL_const
        print "log likelihood:"
        print self.LL(x_batch, y_batch) 
        
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
    data = 'bh_50'
    data_suffix = '_tr_1.csv'
    data_dir = '../../data/uci/'
    data_prefix = data_dir + data
    x_tr, y_tr = input_tools.get_x_y_tr_data(data_prefix, data_suffix)
    x_tr = np.genfromtxt('../../data/linear_input_data.txt', delimiter = ',')
    y_tr = x_tr
    batch_size = x_tr.shape[0]
    batch_size = x_tr.shape[0]
    ###### get weight information
    weights_dir = '../../data/' #for forward test
    a1_size = 0
    num_inputs = tools.get_num_inputs(x_tr)
    num_outputs = tools.get_num_outputs(y_tr)
    layer_sizes = [1, num_inputs] * 2 #if using slp, leave this list empty
    m_trainable_arr = [True, True] * 2 + [False]
    b_trainable_arr = [True, True] * 2 + [False]	
    num_weights = tools.calc_num_weights3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
    ###### check shapes of training data
    x_tr, y_tr = tools.reshape_x_y_twod(x_tr, y_tr)
    ###### setup prior
    hyper_type = "deterministic" # "stochastic" or "deterministic"
    var_type = "deterministic" # "stochastic" or "deterministic"
    weight_shapes = tools.get_weight_shapes3(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr)
    dependence_lengths = tools.get_degen_dependence_lengths(weight_shapes, independent = True)
    if hyper_type == "deterministic" and var_type == "deterministic":
        prior_types = [4]
        prior_hyperparams = [[0., 1.]]
        param_prior_types = [0]
        prior = inverse_priors.inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, num_weights)
        n_stoc = 0
        n_stoc_var = 0
    elif hyper_type == "stochastic" and var_type == "deterministic":
        granularity = 'single'
        hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
        hyperprior_types = [9]
        prior_types = [4]
        hyperprior_params = [[1. / 2., 1. / (2. * 100)]]
        prior_hyperparams = [0.]
        param_hyperprior_types = [0]
        param_prior_types = [0]
        n_stoc = len(hyper_dependence_lengths)
        prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, num_weights)
        n_stoc_var = 0
    elif hyper_type == "stochastic" and var_type == "stochastic":
        granularity = 'single'
        hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
        var_dependence_lengths = [1]
        n_stoc_var = len(var_dependence_lengths)
        hyperprior_types = [9]
        var_prior_types = [10]
        prior_types = [4]
        hyperprior_params = [[1. / 2., 1. / (2. * 100)]]
        var_prior_params = [[1. / 2., 1. / (2. * 100)]]
        prior_hyperparams = [0.]
        param_hyperprior_types = [0]
        var_param_prior_types = [0]
        param_prior_types = [0]
        n_stoc = len(hyper_dependence_lengths)
        prior = isvp.inverse_stoc_var_hyper_prior(hyperprior_types, var_prior_types, prior_types, hyperprior_params, var_prior_params, prior_hyperparams, hyper_dependence_lengths, var_dependence_lengths, dependence_lengths, param_hyperprior_types, var_param_prior_types, param_prior_types, n_stoc, n_stoc_var, num_weights)
###### test prior output from nn setup
    if "nn_prior_test" in run_string:
        prior_tests.nn_prior_test(prior, n_stoc + n_stoc_var + num_weights)
    ###### setup keras model
    model = kms.mlp_ResNet_2(num_inputs, num_outputs, layer_sizes)
    km = keras_model(model, x_tr, y_tr, batch_size, n_stoc_var)
    loss = 'squared_error' # 'squared_error', 'av_squared_error', 'categorical_crossentropy', 'av_categorical_crossentropy'
    km.setup_LL(loss)
    ###### test llhood output
    if "forward_test_linear" in run_string:
        forward_tests.forward_test_linear([km], n_stoc_var + num_weights, weights_dir)
    ###### setup polychord
    nDerived = 0
    settings = PyPolyChord.settings.PolyChordSettings(n_stoc + n_stoc_var + num_weights, nDerived)
    settings.base_dir = './keras_chains/'
    settings.file_root = data + "_sh_sv_slp_1"
    settings.nlive = 1000
    ###### run polychord
    if "polychord1" in run_string:
        PyPolyChord.run_polychord(km, n_stoc, n_stoc_var, num_weights, nDerived, settings, prior, polychord_tools.dumper)
    if "writeparamnames" in run_string:
        output_tools.write_paramnames(num_inputs, layer_sizes, num_outputs, m_trainable_arr, b_trainable_arr, 'bh_50_slp_1', False, False, True, False)

if __name__ == '__main__':
	run_string = 'forward_test_linear'
	main(run_string)


    
        

