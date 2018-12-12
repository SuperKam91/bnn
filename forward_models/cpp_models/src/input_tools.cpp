/* external codebase */
#include <vector>
#include <fstream>

/* in-house code */
#include "input_tools.hpp"

std::vector<double> get_w_vec_from_file(const uint & num_weights, const std::string & weight_path) { 
    std::vector<double> weight_v = column_file_2_vec(num_weights, weight_path);
    return weight_v;
}

std::vector<double> column_file_2_vec(const uint & num_data, const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    values.reserve(num_data);
    uint i = 0;
    while ((std::getline(indata, line)) && (i < num_data)) {
        values.push_back(std::stod(line));
        ++i;
    }
    indata.close();
    return values;
}
