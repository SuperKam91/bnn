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
#include "mathematics.hpp"
#include "prior_tests.hpp"

//extern and global variables
//g's denote global
//e's denote external (global)
//could write functions to calculate n_inputs/outputs and m from data files,
//but easier to do this in python and fill in values here manually
//data parameters
//-----------------------------------------------------------------------------
const uint g_n_inputs = 14;
const uint g_n_outputs = 2;
const uint g_m = 48;
const uint g_b_size = 48; 
//-----------------------------------------------------------------------------
//nn parameters
//-----------------------------------------------------------------------------
const std::vector<uint> g_l_sizes = {};
const std::vector<bool> g_trainable_w_v = {true};
const std::vector<bool> g_trainable_b_v = {true};
//external as it is needed by forward_test and polychord_interface
const uint e_n_weights = calc_num_weights(g_n_inputs, g_l_sizes, g_n_outputs, g_trainable_w_v, g_trainable_b_v);
std::vector<uint> g_weight_shapes = get_weight_shapes(g_n_inputs, g_l_sizes, g_n_outputs, g_trainable_w_v, g_trainable_b_v);
#define g_NEURAL_NETWORK slp_sm 
//-----------------------------------------------------------------------------
//i/o files. see polychord_interfaces.cpp for specifying chains file_root
//----------------------------------------------------------------------------
const std::string e_data_dir = "../../data/kaggle/";
const std::string e_data = "FIFA_2018_Statistics";
const std::string g_x_path = e_data_dir + e_data + "_x_tr_10.csv";
const std::string g_y_path = e_data_dir + e_data + "_y_tr_10.csv";
const std::string e_chains_dir = "./cpp_chains/";
const std::string e_weights_dir = "../../data/";
//create nn forward_prop class object
//-----------------------------------------------------------------------------
//this initialisation has to be done at global scope in some file, unless polychord wrapper is modified to take forward_prop object.
//could declare here then give definition in a function, but would have to get operator= to work
forward_prop e_nn(g_n_inputs, g_n_outputs, g_m, g_b_size, g_l_sizes, g_x_path, g_y_path, g_NEURAL_NETWORK, g_trainable_w_v, g_trainable_b_v);
//prior setup
//-----------------------------------------------------------------------------
bool g_independent = true;
std::vector<uint> g_dependence_lengths = get_degen_dependence_lengths(g_weight_shapes, g_independent);
//the following parameters should be specified manually
//-----------------------------------------------------------------------------
std::vector<uint> g_prior_types = {4};
std::vector<double> g_prior_hyperparams = {0., 1.};
std::vector<uint> g_param_prior_types = {0};
inverse_prior e_ip(g_prior_types, g_prior_hyperparams, g_dependence_lengths, g_param_prior_types, e_n_weights);
//-----------------------------------------------------------------------------

int main() {
	bool nn_prior_t = false;
	bool forward_t_linear = false;
	bool polychord1_run = true;
	bool profiling = false;
	//setup ll, can't do this outside of function
	e_nn.setup_LL("categorical_crossentropy"); // gauss, av_gauss, categorical_crossentropy, av_categorical_crossentropy, dummy
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> finish;
	if (profiling) {
		start = std::chrono::high_resolution_clock::now();
	}
	if (nn_prior_t) {
		nn_prior_test(!profiling); //suppress output
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



