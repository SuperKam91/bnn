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

const uint g_n_inputs = 2;
const uint g_n_outputs = 2;
const uint g_m = 6;
const uint g_b_size = 6; 
const std::vector<uint> g_l_sizes = {5};
const std::string g_x_path = "./data/scratch_gauss_slp_x.txt";
const std::string g_y_path = "./data/scratch_gauss_slp_y.txt";
#define NEURAL_NETWORK slp_nn_1 //only has file scope
//this initialisation has to be done at global scope in some file.
//could declare here then give definition in a function, but would have to get operator= to work
forward_prop g_slp_nn(g_n_inputs, g_n_outputs, g_m, g_b_size, g_l_sizes, g_x_path, g_y_path, NEURAL_NETWORK);

int main() {
	//setup ll
	g_slp_nn.setup_LL("gauss");
	forward_test_linear();
	return 0;
}


