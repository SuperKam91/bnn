/* external codebases */
#include <vector>
#include <iostream>
#include <Eigen/Dense>

/* in-house code */
#include "forward.hpp"


int main() {

  // generate weights manually for test. not needed in production
  //---------------------------------------------------------------------------
  const uint num_inputs = 2;
  const uint num_outputs = 2;
  const std::vector<uint> layer_sizes = std::vector<uint>{5};
  const uint num_weights = calc_num_weights(num_inputs, layer_sizes, num_outputs);
  Eigen::VectorXd w = Eigen::VectorXd::LinSpaced(num_weights, 0, num_weights - 1);
  //---------------------------------------------------------------------------
  forward_prop slp_nn(2, 2, 6, 6, std::vector<uint>{5}, "./data/scratch_gauss_slp_x.txt", "./data/scratch_gauss_slp_y.txt");
  slp_nn.calc_LL_norm("gauss");
  slp_nn(w);
  return 0;
}