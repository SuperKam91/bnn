#for some reason if you import scipy.stats before tf, get ImportError on scipy.stats import
import inverse_priors
import inverse_stoc_hyper_priors as isp

#########commercial modules

#in-house modules
import keras_models as kms
import keras_forward as kf
import np_models as npms
import np_forward as npf
import tf_graphs as tfgs
import tf_forward as tff
import tools
import PyPolyChord
import PyPolyChord.settings
import polychord_tools
import input_tools
import forward_tests
import prior_tests

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
    km = kf.keras_model(model, x_tr, y_tr, batch_size)
    loss = 'mse' # 'squared_error', 'av_squared_error', 'categorical_crossentropy', 'av_categorical_crossentropy'
    km.setup_LL(loss)
    #setup tf graph
    tf_graph = tfgs.slp_graph
    tfm = tff.tf_model(tf_graph, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr, b_trainable_arr)
    fit_metric = 'chisq' # 'chisq', 'av_chisq', 'categorical_crossentropy', 'av_categorical_crossentropy'
    tfm.setup_LL(fit_metric)
    #set up np model
    np_nn = npms.slp_nn
    npm = npf.np_model(np_nn, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr, b_trainable_arr)
    ll_type = 'gauss' # 'gauss', 'av_gauss', 'categorical_crossentropy', 'av_categorical_crossentropy'
    npm.setup_LL(ll_type)
    ###### test llhood output
    if "k_forward_test_linear" in run_string:
        forward_tests.forward_test_linear([km], num_weights, data_dir)
    if "tf_forward_test_linear" in run_string:
        forward_tests.forward_test_linear([tfm], num_weights, data_dir)
    if "np_forward_test_linear" in run_string:
        forward_tests.forward_test_linear([npm], num_weights, data_dir)
    ###### setup prior
    if hyper_type == "deterministic":
        prior_types = [4]
        prior_hyperparams = [[0., 1.]]
        param_prior_types = [0]
        prior = inverse_priors.inverse_prior(prior_types, prior_hyperparams, dependence_lengths, param_prior_types, num_weights)
        n_stoc = 0
    elif hyper_type == "stochastic":
        granularity = 'single'
        hyper_dependence_lengths = tools.get_hyper_dependence_lengths(weight_shapes, granularity)
        hyperprior_types = [9]
        prior_types = [4]
        hyperprior_params = [[0.1 / 2., 0.1 / (2. * 100)]]
        prior_hyperparams = [0.]
        param_hyperprior_types = [0]
        param_prior_types = [0]
        n_stoc = len(hyper_dependence_lengths)
        prior = isp.inverse_stoc_hyper_prior(hyperprior_types, prior_types, hyperprior_params, prior_hyperparams, hyper_dependence_lengths, dependence_lengths, param_hyperprior_types, param_prior_types, n_stoc, num_weights)
    ###### test prior output from nn setup
    if "nn_prior_test" in run_string:
        prior_tests.nn_prior_test(prior, n_stoc + num_weights)
    ###### setup polychord
    nDerived = 0
    settings = PyPolyChord.settings.PolyChordSettings(n_stoc + num_weights, nDerived)
    settings.file_root = data + "slp_1"
    settings.nlive = 1000
    ###### run polychord
    if "k_polychord1" in run_string:
        settings.base_dir = './keras_chains/'
        PyPolyChord.run_polychord(km, num_weights, nDerived, settings, prior, polychord_tools.dumper)
    if "tf_polychord1" in run_string:
        settings.base_dir = './tf_chains/'
        PyPolyChord.run_polychord(tfm, num_weights, nDerived, settings, prior, polychord_tools.dumper)
    if "np_polychord1" in run_string:
        settings.base_dir = './np_chains/'
        PyPolyChord.run_polychord(npm, num_weights, nDerived, settings, prior, polychord_tools.dumper)

if __name__ == '__main__':
    run_string = ''
    main(run_string)
