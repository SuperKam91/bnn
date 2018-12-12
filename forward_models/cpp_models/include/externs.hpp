#pragma once

/* external codebases */
#include <string>

/* in-house code */
#include "forward.hpp"
#include "inverse_priors.hpp"

extern forward_prop e_nn;
extern const uint e_n_weights;
extern const std::string e_data_dir;
extern const std::string e_data;
extern const std::string e_chains_dir;
extern const std::string e_weights_dir;
extern inverse_prior e_ip;