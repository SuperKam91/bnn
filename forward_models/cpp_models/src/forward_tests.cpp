/* external codebases */
#include <Eigen/Dense>

/* in-house code */
#include "forward_tests.hpp"
#include "tools.hpp"
#include "externs.hpp"

void forward_test_linear(){
	const uint num_weights = calc_num_weights(g_n_inputs, g_l_sizes, g_n_outputs);
    Eigen::VectorXd w = Eigen::VectorXd::LinSpaced(num_weights, 0, num_weights - 1);
    std::cout << g_slp_nn(w) << std::endl;
}