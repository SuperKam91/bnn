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
import inverse_priors
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
    loss = 'mse' 
    km.setup_LL(loss)
    #setup tf graph
    tf_graph = tfgs.slp_graph
    tfm = tff.tf_model(tf_graph, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr, b_trainable_arr)
    fit_metric = 'chisq'
    tfm.setup_LL(fit_metric)
    #set up np model
    np_nn = npms.slp_nn
    npm = npf.np_model(np_nn, x_tr, y_tr, batch_size, layer_sizes, m_trainable_arr, b_trainable_arr)
    ll_type = 'gauss'
    npm.setup_LL(ll_type)
    ###### test llhood output
    if "k_forward_test_linear" in run_string:
        forward_tests.forward_test_linear([km])
    if "tf_forward_test_linear" in run_string:
        forward_tests.forward_test_linear([tfm])
    if "np_forward_test_linear" in run_string:
        forward_tests.forward_test_linear([npm])
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
    settings.file_root = data
    settings.nlive = 200
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
