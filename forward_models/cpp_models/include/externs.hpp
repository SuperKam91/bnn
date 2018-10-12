#pragma once

#include <vector>
#include <string>

#include "forward.hpp"
#include "nn_models.hpp"

//g's denote global scope
extern forward_prop g_slp_nn;
extern const uint g_n_inputs;
extern const uint g_n_outputs;
extern const std::vector<uint> g_l_sizes;
extern const uint g_m;
extern const uint g_b_size;
extern const std::string g_x_path;
extern const std::string g_y_path;