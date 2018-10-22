/* external codebase */
#include <vector>

/* in-house code */
#include <tools.hpp>

//not used typedef uint here as it requires including the relevant header, which I can't find
//(other than iostream, which seems inappropriate just for a typedef)

unsigned int calc_num_weights(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs) {
    unsigned int n = (num_inps + 1) * layer_sizes.front();
    for (unsigned int i = 1; i < layer_sizes.size(); ++i) {
        n += (layer_sizes[i-1] + 1) * layer_sizes[i];
    }
    n += (layer_sizes.back() + 1) * num_outs;
    return n;
}

unsigned int calc_num_weights(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs, const std::vector<bool> & trainable_v) {
	unsigned int n = 0;
	if (trainable_v.at(0)) {
		n += (num_inps + 1) * layer_sizes.at(0);
	}
	for (unsigned int i = 1; i < layer_sizes.size(); ++i) {
		if (trainable_v.at(i)) {
			n += (layer_sizes.at(i - 1) + 1) * layer_sizes.at(i);
		}
	}
	if (trainable_v.back()) {
		n += (layer_sizes.back() + 1) * num_outs;
	}
	return n;
}

unsigned int calc_num_weights(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs, const std::vector<bool> & trainable_w_v, const std::vector<bool> & trainable_b_v) {
	unsigned int n = 0;
	if (trainable_w_v.at(0)){
		n += num_inps * layer_sizes.at(0);
	}
	if (trainable_b_v.at(0)){
		n += layer_sizes.at(0);
	}
	for (unsigned int i = 1; i < layer_sizes.size(); ++i) {
		if (trainable_w_v.at(i)) {	
			n += layer_sizes.at(i - 1) * layer_sizes.at(i);
		}
		if (trainable_b_v.at(i)) {
			n += layer_sizes.at(i);
		}
	}
	if (trainable_w_v.back()) {
		n += layer_sizes.back() * num_outs;
	}
	if (trainable_b_v.back()) {
		n += num_outs;
	}
	return n;
}

//more or less copied from forward_prop::get_weight_shapes
std::vector<unsigned int> get_weight_shapes(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs) { 
	std::vector<unsigned int> weight_s;
    weight_s.reserve((layer_sizes.size() + 1) * 2); 
    unsigned int w_rows = num_inps;
    weight_s.push_back(w_rows);
    unsigned int w_cols = layer_sizes.front();
    weight_s.push_back(w_cols);
    unsigned int b_rows = w_cols; 
    weight_s.push_back(b_rows);
    unsigned int b_cols = 1;
    weight_s.push_back(b_cols);
    for (unsigned int i = 1; i < layer_sizes.size(); ++i) {
        w_rows = w_cols;
        weight_s.push_back(w_rows);
        w_cols = layer_sizes.at(i);
        weight_s.push_back(w_cols);
        b_rows = w_cols;
        weight_s.push_back(b_rows);
        weight_s.push_back(b_cols);        
    }
    w_rows = w_cols;
    weight_s.push_back(w_rows);
    w_cols = num_outs;
    weight_s.push_back(w_cols);
    b_rows = w_cols;
    weight_s.push_back(b_rows);
    weight_s.push_back(b_cols);
    return weight_s;
}

std::vector<unsigned int> get_weight_shapes(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs, const std::vector<bool> & trainable_v) {
	std::vector<unsigned int> weight_s;
	//don't bother reserving, could count number of trues in bool arr, but cba
	unsigned int w_rows = num_inps;
	unsigned int w_cols;
    unsigned int b_cols = 1;
    unsigned int b_rows;
	for (unsigned int i = 0; i < layer_sizes.size(); ++i) {
		w_cols = layer_sizes.at(i);
		b_rows = w_cols;
		if (trainable_v.at(i)) {	
	        weight_s.push_back(w_rows);
	        weight_s.push_back(w_cols);
	        weight_s.push_back(b_rows);
	        weight_s.push_back(b_cols);
		}
		w_rows = w_cols;
	}
	if (trainable_v.back()) {
		w_rows = w_cols;
	    weight_s.push_back(w_rows);
	    w_cols = num_outs;
	    weight_s.push_back(w_cols);
	    b_rows = w_cols;
	    weight_s.push_back(b_rows);
	    weight_s.push_back(b_cols);
	}
	return weight_s;
}

std::vector<unsigned int> get_weight_shapes(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs, const std::vector<bool> & trainable_w_v, const std::vector<bool> & trainable_b_v) {
	std::vector<unsigned int> weight_s;
	unsigned int w_rows = num_inps;
	unsigned int w_cols;
    unsigned int b_cols = 1;
    unsigned int b_rows;
	for (unsigned int i = 0; i < layer_sizes.size(); ++i) {
		w_cols = layer_sizes.at(i);
		b_rows = w_cols;
		if (trainable_w_v.at(i)) {	
	        weight_s.push_back(w_rows);
	        weight_s.push_back(w_cols);
	    }
	    if (trainable_b_v.at(i)) {   
	        weight_s.push_back(b_rows);
	        weight_s.push_back(b_cols);
		}
		w_rows = w_cols;
	}
	if (trainable_w_v.back()) {
		w_rows = w_cols;
	    weight_s.push_back(w_rows);
	    w_cols = num_outs;
	    weight_s.push_back(w_cols);
	}
	if (trainable_b_v.back()) {
	    b_rows = w_cols;
	    weight_s.push_back(b_rows);
	    weight_s.push_back(b_cols);
	}
	return weight_s;	
}