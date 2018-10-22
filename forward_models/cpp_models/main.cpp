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

//extern and global variables
//g's denote global
//e's denote external (global)
//could write functions to calculate n_inputs/outputs and m from data files,
//but easier to do this in python and fill in values here manually
const uint g_n_inputs = 1;
const uint g_n_outputs = 1;
const uint g_m = 1000;
const uint g_b_size = 1000; 
const std::vector<uint> g_l_sizes = {2};
const std::vector<bool> g_trainable_w_v = {true, false};
const std::vector<bool> g_trainable_b_v = {false, false};
const std::string e_data_dir = "../../data/";
const std::string e_data = "simple_tanh";
const std::string g_x_path = e_data_dir + e_data + "_x.txt";
const std::string g_y_path = e_data_dir + e_data + "_y.txt";
const std::string e_chains_dir = "./cpp_chains/";
//external as it is needed by forward_test and polychord_interface
const uint e_n_weights = calc_num_weights(g_n_inputs, g_l_sizes, g_n_outputs, g_trainable_w_v, g_trainable_b_v);
#define NEURAL_NETWORK slp_nn_1 //only has file scope
//this initialisation has to be done at global scope in some file, unless polychord wrapper is modified to take forward_prop object.
//could declare here then give definition in a function, but would have to get operator= to work
forward_prop e_slp_nn(g_n_inputs, g_n_outputs, g_m, g_b_size, g_l_sizes, g_x_path, g_y_path, NEURAL_NETWORK, g_trainable_w_v, g_trainable_b_v);

int main() {
	//setup ll, can't do this outside of function
	e_slp_nn.setup_LL("gauss");
	run_polychord_wrap();
	return 0;
}


