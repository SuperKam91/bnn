/* external codebase */
#include <vector>

/* in-house code */
#include <tools.hpp>

//only used when weights need to be generated manually
uint calc_num_weights(const uint & num_inps, const std::vector<uint> & layer_sizes, const uint & num_outs) {
    uint n = (num_inps + 1) * layer_sizes.front();
    for (uint i = 1; i < layer_sizes.size(); ++i) {
        n += (layer_sizes[i-1] + 1) * layer_sizes[i];
    }
    n += (layer_sizes.back() + 1) * num_outs;
    return n;
}