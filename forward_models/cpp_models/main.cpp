/* external codebases */
#include <string>
#include <vector>
#include <iostream>

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
const uint g_n_inputs = 1;
const uint g_n_outputs = 1;
const uint g_m = 1000;
const uint g_b_size = 1000; 
//-----------------------------------------------------------------------------
//nn parameters
//-----------------------------------------------------------------------------
const std::vector<uint> g_l_sizes = {2};
const std::vector<bool> g_trainable_w_v = {true, false};
const std::vector<bool> g_trainable_b_v = {false, false};
//external as it is needed by forward_test and polychord_interface
const uint e_n_weights = calc_num_weights(g_n_inputs, g_l_sizes, g_n_outputs, g_trainable_w_v, g_trainable_b_v);
std::vector<uint> g_weight_shapes = get_weight_shapes(g_n_inputs, g_l_sizes, g_n_outputs, g_trainable_w_v, g_trainable_b_v);
#define g_NEURAL_NETWORK slp_nn_1 
//-----------------------------------------------------------------------------
//i/o files
//-----------------------------------------------------------------------------
const std::string e_data_dir = "../../data/";
const std::string e_data = "simple_tanh";
const std::string g_x_path = e_data_dir + e_data + "_x.txt";
const std::string g_y_path = e_data_dir + e_data + "_y.txt";
const std::string e_chains_dir = "./cpp_chains/";
//create nn forward_prop class object
//-----------------------------------------------------------------------------
//this initialisation has to be done at global scope in some file, unless polychord wrapper is modified to take forward_prop object.
//could declare here then give definition in a function, but would have to get operator= to work
forward_prop e_slp_nn(g_n_inputs, g_n_outputs, g_m, g_b_size, g_l_sizes, g_x_path, g_y_path, g_NEURAL_NETWORK, g_trainable_w_v, g_trainable_b_v);
//prior setup
//-----------------------------------------------------------------------------
bool g_independent = false;
std::vector<uint> g_dependence_lengths = get_degen_dependence_lengths(g_weight_shapes, g_independent);
//the following parameters should be specified manually
//-----------------------------------------------------------------------------
std::vector<uint> g_prior_types = {7};
std::vector<double> g_prior_hyperparams = {-2., 2.};
std::vector<uint> g_param_prior_types = {0};
inverse_prior e_ip(g_prior_types, g_prior_hyperparams, g_dependence_lengths, g_param_prior_types, e_n_weights);
//-----------------------------------------------------------------------------

int main() {
	bool nn_prior_t = false;
	bool forward_t_linear = false;
	bool polychord1_run = false;
	bool print_out = true;
	//setup ll, can't do this outside of function
	e_slp_nn.setup_LL("gauss");
	if (nn_prior_t) {
		nn_prior_test(print_out);
	}	
	if (forward_t_linear) {
		forward_test_linear(print_out);
	}
	if (polychord1_run) {
		run_polychord_wrap();
	}
	return 0;
	}



