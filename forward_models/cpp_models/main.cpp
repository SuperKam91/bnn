/* external codebases */
#include <string>
#include <vector>
#include <iostream>
#include <chrono>  // for high_resolution_clock for timing

/* in-house code */
#include "forward.hpp"
#include "externs.hpp"
#include "polychord_interfaces.hpp"
#include "nn_models.hpp"
#include "forward_tests.hpp"
#include "input_tools.hpp"
#include "tools.hpp"
#include "inverse_priors.hpp"
#include "inverse_stoc_hyper_priors.hpp"
#include "inverse_stoc_var_hyper_priors.hpp"
#include "mathematics.hpp"
#include "prior_tests.hpp"

//extern and global variables
//g's denote global
//e's denote external (global)
//could write functions to calculate n_inputs/outputs and m from data files,
//but easier to do this in python and fill in values here manually
//data parameters
//-----------------------------------------------------------------------------
const uint g_n_inputs = 8;
const uint g_n_outputs = 1;
const uint g_m = 3000;
const uint g_b_size = 3000; 
//-----------------------------------------------------------------------------
//nn parameters
//-----------------------------------------------------------------------------
const std::vector<uint> g_l_sizes = {8};
const std::vector<bool> g_trainable_w_v = {true, true};
const std::vector<bool> g_trainable_b_v = {true, true};
//external as it is needed by forward_test and polychord_interface
const uint e_n_weights = calc_num_weights(g_n_inputs, g_l_sizes, g_n_outputs, g_trainable_w_v, g_trainable_b_v);
std::vector<uint> g_weight_shapes = get_weight_shapes(g_n_inputs, g_l_sizes, g_n_outputs, g_trainable_w_v, g_trainable_b_v);
#define g_NEURAL_NETWORK mlp_tanh_1
//prior setup
//-----------------------------------------------------------------------------
bool g_independent = false;
std::vector<uint> g_dependence_lengths = get_degen_dependence_lengths(g_weight_shapes, g_independent);
//the following parameters should be specified manually
//-----------------------------------------------------------------------------
//first determine whether to use stochastic or deterministic hyperparameters
//-----------------------------------------------------------------------------
#define g_PRIOR_TYPE 'D' //'D' for deterministic or 'S' for stochastic
//-----------------------------------------------------------------------------
//number of stochastic likelihood variances only currently supports 0 or 1
//----------------------------------------------------------------------------
#define g_VAR_TYPE 'D' //'D' for deterministic or 'S' for stochastic
//----------------------------------------------------------------------------
#if ((g_PRIOR_TYPE == 'D') && (g_VAR_TYPE == 'D'))
	const std::string e_hyper_type = "deterministic"; //"deterministic" or "stochastic"
	const std::string e_var_type = "deterministic"; //"deterministic" or "stochastic"
	std::vector<uint> g_prior_types = {4, 15};
	std::vector<double> g_prior_hyperparams = {0., 1., 0., 1.};
	std::vector<uint> g_param_prior_types = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inverse_prior e_ip(g_prior_types, g_prior_hyperparams, g_dependence_lengths, g_param_prior_types, e_n_weights);
	//DO NOT CHANGE FOLLOWING LINES. e_sh_ip CLASS IS CREATED PURELY TO MAKE THE CODE COMPILE, IT WILL NEVER BE USED
	//IN THIS CASE
	//--------------------------------------------------------------------------------------------------------------
	const uint e_n_stoc = 0; //even though not used, because it is extern, still needs to be declared
	std::vector<uint> DUMMY_hyperprior_types = {};
	std::vector<uint> DUMMY_prior_types = {};
	std::vector<double> DUMMY_hyperprior_params = {};
	std::vector<double> DUMMY_prior_hyperparams = {};
	std::vector<uint> DUMMY_hyper_dependence_lengths = {};
	std::vector<uint> DUMMY_dependence_lengths = {};
	std::vector<uint> DUMMY_param_hyperprior_types = {};
	std::vector<uint> DUMMY_param_prior_types = {};
	sh_inverse_prior e_sh_ip(DUMMY_hyperprior_types, DUMMY_prior_types, DUMMY_hyperprior_params, DUMMY_prior_hyperparams, DUMMY_hyper_dependence_lengths, DUMMY_dependence_lengths, DUMMY_param_hyperprior_types, DUMMY_param_prior_types, 0, 0);
	//DO NOT CHANGE FOLLOWING LINES. e_svh_ip CLASS IS CREATED PURELY TO MAKE THE CODE COMPILE, IT WILL NEVER BE USED
	//IN THIS CASE
	//--------------------------------------------------------------------------------------------------------------
	const uint e_n_stoc_var = 0; //ditto
	std::vector<uint> DUMMY_var_prior_types = {};
	std::vector<double> DUMMY_var_prior_params = {};
	std::vector<uint> DUMMY_var_dependence_lengths = {};
	std::vector<uint> DUMMY_var_param_prior_types = {};
	svh_inverse_prior e_svh_ip(DUMMY_hyperprior_types, DUMMY_var_prior_types, DUMMY_prior_types, DUMMY_hyperprior_params, DUMMY_var_prior_params, DUMMY_prior_hyperparams, DUMMY_hyper_dependence_lengths, DUMMY_var_dependence_lengths, DUMMY_dependence_lengths, DUMMY_param_hyperprior_types, DUMMY_var_param_prior_types, DUMMY_param_prior_types, 0, 0, 0);
	//--------------------------------------------------------------------------------------------------------------
#elif ((g_PRIOR_TYPE == 'S') && (g_VAR_TYPE == 'D'))
	const std::string e_hyper_type = "stochastic"; //"deterministic" or "stochastic"
	const std::string e_var_type = "deterministic"; //"deterministic" or "stochastic"
	std::string g_granularity = "single"; //"single", "layer" or "input_size"
	std::vector<uint> g_hyper_dependence_lengths = get_hyper_dependence_lengths(g_weight_shapes, g_granularity);
	const uint e_n_stoc = static_cast<uint>(g_hyper_dependence_lengths.size());
	std::vector<uint> g_hyperprior_types = {9};
	std::vector<uint> g_prior_types = {4};
	std::vector<double> g_hyperprior_params = {2. / 2., 2. / (2. * 1.)};
	std::vector<double> g_prior_hyperparams = {0.};
	std::vector<uint> g_param_hyperprior_types = {0};
	std::vector<uint> g_param_prior_types = {0};
	sh_inverse_prior e_sh_ip(g_hyperprior_types, g_prior_types, g_hyperprior_params, g_prior_hyperparams, g_hyper_dependence_lengths, g_dependence_lengths, g_param_hyperprior_types, g_param_prior_types, e_n_stoc, e_n_weights);
	//DO NOT CHANGE FOLLOWING LINES. e_ip CLASS IS CREATED PURELY TO MAKE THE CODE COMPILE, IT WILL NEVER BE USED
	//IN THIS CASE
	//--------------------------------------------------------------------------------------------------------------
	std::vector<uint> DUMMY_prior_types = {};
	std::vector<double> DUMMY_prior_hyperparams = {};
	std::vector<uint> DUMMY_dependence_lengths = {};
	std::vector<uint> DUMMY_param_prior_types = {};
	inverse_prior e_ip(DUMMY_prior_types, DUMMY_prior_hyperparams, DUMMY_dependence_lengths, DUMMY_param_prior_types, 0);
	//DO NOT CHANGE FOLLOWING LINES. e_svh_ip CLASS IS CREATED PURELY TO MAKE THE CODE COMPILE, IT WILL NEVER BE USED
	//IN THIS CASE
	//--------------------------------------------------------------------------------------------------------------
	const uint e_n_stoc_var = 0; //ditto
	std::vector<uint> DUMMY_hyperprior_types = {};
	std::vector<uint> DUMMY_var_prior_types = {};
	std::vector<double> DUMMY_hyperprior_params = {};
	std::vector<double> DUMMY_var_prior_params = {};
	std::vector<uint> DUMMY_hyper_dependence_lengths = {};
	std::vector<uint> DUMMY_var_dependence_lengths = {};
	std::vector<uint> DUMMY_param_hyperprior_types = {};
	std::vector<uint> DUMMY_var_param_prior_types = {};
	svh_inverse_prior e_svh_ip(DUMMY_hyperprior_types, DUMMY_var_prior_types, DUMMY_prior_types, DUMMY_hyperprior_params, DUMMY_var_prior_params, DUMMY_prior_hyperparams, DUMMY_hyper_dependence_lengths, DUMMY_var_dependence_lengths, DUMMY_dependence_lengths, DUMMY_param_hyperprior_types, DUMMY_var_param_prior_types, DUMMY_param_prior_types, 0, 0, 0);
	//--------------------------------------------------------------------------------------------------------------
#elif ((g_PRIOR_TYPE == 'S') && (g_VAR_TYPE == 'S'))
	const std::string e_hyper_type = "stochastic"; //"deterministic" or "stochastic"
	const std::string e_var_type = "stochastic"; //"deterministic" or "stochastic"
	std::string g_granularity = "single"; //"single", "layer" or "input_size"
	// std::string g_granularity = "layer";
	// std::string g_granularity = "input_size";
	//std::vector<uint> g_hyper_dependence_lengths = get_hyper_dependence_lengths(g_weight_shapes, g_granularity);
	std::vector<uint> g_hyper_dependence_lengths = {1}; //single
	// std::vector<uint> g_hyper_dependence_lengths = {72, 9}; //layer
	// std::vector<uint> g_hyper_dependence_lengths = {8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1}; //input_size
	std::vector<uint> g_var_dependence_lengths = {1};
	const uint e_n_stoc = static_cast<uint>(g_hyper_dependence_lengths.size());
	const uint e_n_stoc_var = static_cast<uint>(g_var_dependence_lengths.size());
	std::vector<uint> g_hyperprior_types = {9}; //single
	// std::vector<uint> g_hyperprior_types = {9}; //layer
	// std::vector<uint> g_hyperprior_types = {9, 9}; //input_size
	std::vector<uint> g_var_prior_types = {10};
	std::vector<uint> g_prior_types = {4, 15};
	std::vector<double> g_hyperprior_params = {2. / 2., 2. / (2. * 1.)}; //single
	// std::vector<double> g_hyperprior_params = {2. / 2., 2. / (2. * 1.)}; //layer
	// std::vector<double> g_hyperprior_params = {2. / 2., 2. / (2. * 1.), 2. / 2., 2. / (2. * 1. * static_cast<double>(g_l_sizes.at(0)))}; //input_size
	std::vector<double> g_var_prior_params = {2. / 2., 2. / (2. * 1.)};
	std::vector<double> g_prior_hyperparams = {0}; //single
	// std::vector<double> g_prior_hyperparams = {0.}; //layer
	// std::vector<double> g_prior_hyperparams = {0., 0.}; //input_size
	std::vector<uint> g_param_hyperprior_types = {0}; //single
	// std::vector<uint> g_param_hyperprior_types = {0, 0}; //layer
	// std::vector<uint> g_param_hyperprior_types = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}; //input_size
	std::vector<uint> g_var_param_prior_types = {0};
	std::vector<uint> g_param_prior_types = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	svh_inverse_prior e_svh_ip(g_hyperprior_types, g_var_prior_types, g_prior_types, g_hyperprior_params, g_var_prior_params, g_prior_hyperparams, g_hyper_dependence_lengths, g_var_dependence_lengths, g_dependence_lengths, g_param_hyperprior_types, g_var_param_prior_types, g_param_prior_types, e_n_stoc_var, e_n_stoc, e_n_weights);
	//DO NOT CHANGE FOLLOWING LINES. e_ip CLASS IS CREATED PURELY TO MAKE THE CODE COMPILE, IT WILL NEVER BE USED
	//IN THIS CASE
	//--------------------------------------------------------------------------------------------------------------
	std::vector<uint> DUMMY_prior_types = {};
	std::vector<double> DUMMY_prior_hyperparams = {};
	std::vector<uint> DUMMY_dependence_lengths = {};
	std::vector<uint> DUMMY_param_prior_types = {};
	inverse_prior e_ip(DUMMY_prior_types, DUMMY_prior_hyperparams, DUMMY_dependence_lengths, DUMMY_param_prior_types, 0);
	//DO NOT CHANGE FOLLOWING LINES. e_sh_ip CLASS IS CREATED PURELY TO MAKE THE CODE COMPILE, IT WILL NEVER BE USED
	//IN THIS CASE
	//--------------------------------------------------------------------------------------------------------------
	std::vector<uint> DUMMY_hyperprior_types = {};
	std::vector<double> DUMMY_hyperprior_params = {};
	std::vector<uint> DUMMY_hyper_dependence_lengths = {};
	std::vector<uint> DUMMY_param_hyperprior_types = {};
	sh_inverse_prior e_sh_ip(DUMMY_hyperprior_types, DUMMY_prior_types, DUMMY_hyperprior_params, DUMMY_prior_hyperparams, DUMMY_hyper_dependence_lengths, DUMMY_dependence_lengths, DUMMY_param_hyperprior_types, DUMMY_param_prior_types, 0, 0);
	//--------------------------------------------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------
//i/o files. see polychord_interfaces.cpp for specifying chains file_root
//----------------------------------------------------------------------------
const std::string e_data_dir = "../../data/21cm/";
const std::string e_data = "8_params_21_2";
const std::string g_x_path = e_data_dir + e_data + "_x_phys_3000_tr.csv";
const std::string g_y_path = e_data_dir + e_data + "_y_phys_3000_tr.csv";
const std::string e_chains_dir = "./cpp_chains/";
const std::string e_weights_dir = "../../data/"; //for forward tests
const std::string e_data_suffix = "_phys_mlp_8_3000"; //for chains
// const std::string e_data_suffix = "_phys_sh_sv_mlp_8_300";
// const std::string e_data_suffix = "_phys_lh_sv_mlp_8_300";
// const std::string e_data_suffix = "_phys_ih_sv_mlp_8_300";

//create nn forward_prop class object
//-----------------------------------------------------------------------------
//this initialisation has to be done at global scope in some file, unless polychord wrapper is modified to take forward_prop object.
//could declare here then give definition in a function, but would have to get operator= to work
forward_prop e_nn(g_n_inputs, g_n_outputs, g_m, g_b_size, g_l_sizes, g_x_path, g_y_path, g_NEURAL_NETWORK, g_trainable_w_v, g_trainable_b_v, e_n_weights, e_n_stoc_var);

//-----------------------------------------------------------------------------

int main() {
	bool nn_prior_t = false;
	bool prior_pft = false;
	bool prior_spft = false;
	bool forward_t_linear = false;
	bool polychord1_run = true;
	bool profiling = false; //also dictates whether to print output or not
	//setup ll, can't do this outside of function
	e_nn.setup_LL("gauss"); // gauss, av_gauss, categorical_crossentropy, av_categorical_crossentropy, dummy
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> finish;
	if (profiling) {
		start = std::chrono::high_resolution_clock::now();
	}
	if ((nn_prior_t) && (e_hyper_type == "deterministic")) {
		nn_prior_test(!profiling); 
	}
	else if ((nn_prior_t) && (e_hyper_type == "stochastic")) {
		nn_sh_prior_test(!profiling); 
	}	
	else if ((nn_prior_t) && (e_hyper_type == "stochastic") && (e_var_type == "stochastic")) {
		nn_sv_sh_prior_test(!profiling); 
	}		
	if (prior_pft) {
		prior_functions_test(!profiling);
	}
	if (prior_spft) {
		stoc_prior_functions_test(!profiling);		
	}
	if (forward_t_linear) {
		forward_test_linear();
	}
	if (polychord1_run) {
		run_polychord_wrap(profiling);
	}
	if (profiling) {
		finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "time elapsed was: " << elapsed.count() << std::endl;
	}
	return 0;
}



