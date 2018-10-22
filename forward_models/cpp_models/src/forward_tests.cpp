/* external codebases */
#include <Eigen/Dense>
#include <string>
#include <vector>

/* in-house code */
#include "forward_tests.hpp"
#include "input_tools.hpp"
#include "externs.hpp"

double forward_test_linear(){
    std::string weight_type = "linear";
	std::string weight_file = e_data_dir + weight_type + "_weights.txt";
	std::vector<double> w_v = get_w_vec_from_file(e_n_weights, weight_file);
    Eigen::Map<Eigen::VectorXd> w_m(w_v.data(), e_n_weights);
    return e_slp_nn(w_m);
}