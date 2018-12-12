#pragma once

/* external codebases */
#include <vector>

//without header such as iostream, typedef uint isn't defined, so use unsigned int here (and in tools.cpp) instead.

unsigned int calc_num_weights(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs);

unsigned int calc_num_weights(const unsigned int &, const std::vector<unsigned int> &, const unsigned int &, const std::vector<bool> &);

unsigned int calc_num_weights(const unsigned int &, const std::vector<unsigned int> &, const unsigned int &, const std::vector<bool> &, const std::vector<bool> &);

std::vector<unsigned int> get_weight_shapes(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs);

std::vector<unsigned int> get_weight_shapes(const unsigned int &, const std::vector<unsigned int> &, const unsigned int &, const std::vector<bool> &);

std::vector<unsigned int> get_weight_shapes(const unsigned int &, const std::vector<unsigned int> &, const unsigned int &, const std::vector<bool> &, const std::vector<bool> &);

std::vector<unsigned int> get_degen_dependence_lengths(const std::vector<unsigned int> &, const bool &);

std::vector<unsigned int> get_degen_dependence_lengths2(const std::vector<unsigned int> &, const bool &);

std::vector<unsigned int> get_degen_dependence_lengths3(const std::vector<unsigned int> &, const bool &);

std::vector<unsigned int> get_degen_dependence_lengths4(const std::vector<unsigned int> &, const bool &);

std::vector<double> get_seq_vec(std::vector<double> seq_values, std::vector<unsigned int> seq_lengths);

std::vector<unsigned int> get_seq_vec(std::vector<unsigned int> seq_values, std::vector<unsigned int> seq_lengths);
