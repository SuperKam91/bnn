/* external codebase */
#include <vector>
#include <string>
#include <iostream>

/* in-house code */
#include <tools.hpp>

//not used typedef uint here as it requires including the relevant header, which I can't find
//(other than iostream, which seems inappropriate just for a typedef)

unsigned int calc_num_weights(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs) {
    //no hidden layers case
    if (layer_sizes.size() == 0) {
    	return  (num_inps + 1) * num_outs; 
    }
    unsigned int n = (num_inps + 1) * layer_sizes.front();
    for (unsigned int i = 1; i < layer_sizes.size(); ++i) {
        n += (layer_sizes[i-1] + 1) * layer_sizes[i];
    }
    n += (layer_sizes.back() + 1) * num_outs;
    return n;
}

unsigned int calc_num_weights(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs, const std::vector<bool> & trainable_v) {
	unsigned int n = 0;
    if (layer_sizes.size() == 0) {
    	if (trainable_v.at(0)) {
    		return  (num_inps + 1) * num_outs;
    	}
    	else {
    		return n;
    	} 
    }
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
	if (layer_sizes.size() == 0) {
    	if (trainable_w_v.at(0)) {
    		n += num_inps * num_outs;
    	}
    	if (trainable_b_v.at(0)) {
    		n += num_outs;
    	} 
    	return n;
    }
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
	//no hidden layers case
	if (layer_sizes.size() == 0) {
		weight_s.push_back(num_inps);
		weight_s.push_back(num_outs);
		weight_s.push_back(1);
		weight_s.push_back(num_outs);
		return weight_s;
	}
    weight_s.reserve((layer_sizes.size() + 1) * 2); 
    unsigned int w_rows = num_inps;
    weight_s.push_back(w_rows);
    unsigned int w_cols = layer_sizes.front();
    weight_s.push_back(w_cols);
    unsigned int b_rows = 1;
    weight_s.push_back(b_rows);
    unsigned int b_cols = w_cols; 
    weight_s.push_back(b_cols);
    for (unsigned int i = 1; i < layer_sizes.size(); ++i) {
        w_rows = w_cols;
        weight_s.push_back(w_rows);
        w_cols = layer_sizes.at(i);
        weight_s.push_back(w_cols);
        b_cols = w_cols;
        weight_s.push_back(b_rows);
        weight_s.push_back(b_cols);        
    }
    w_rows = w_cols;
    weight_s.push_back(w_rows);
    w_cols = num_outs;
    weight_s.push_back(w_cols);
    b_cols = w_cols;
    weight_s.push_back(b_rows);
    weight_s.push_back(b_cols);
    return weight_s;
}

std::vector<unsigned int> get_weight_shapes(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs, const std::vector<bool> & trainable_v) {
	std::vector<unsigned int> weight_s;
	if (layer_sizes.size() == 0) {
		if (trainable_v.at(0)) {
			weight_s.push_back(num_inps);
			weight_s.push_back(num_outs);
			weight_s.push_back(1);
			weight_s.push_back(num_outs);
		}
		return weight_s;
	}
	//don't bother reserving, could count number of trues in bool arr, but cba
	unsigned int w_rows = num_inps;
	unsigned int w_cols;
    unsigned int b_rows = 1;
    unsigned int b_cols;
	for (unsigned int i = 0; i < layer_sizes.size(); ++i) {
		w_cols = layer_sizes.at(i);
		b_cols = w_cols;
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
	    b_cols = w_cols;
	    weight_s.push_back(b_rows);
	    weight_s.push_back(b_cols);
	}
	return weight_s;
}

std::vector<unsigned int> get_weight_shapes(const unsigned int & num_inps, const std::vector<unsigned int> & layer_sizes, const unsigned int & num_outs, const std::vector<bool> & trainable_w_v, const std::vector<bool> & trainable_b_v) {
	std::vector<unsigned int> weight_s;
	if (layer_sizes.size() == 0) {
		if (trainable_w_v.at(0)) {
			weight_s.push_back(num_inps);
			weight_s.push_back(num_outs);
		}
		if (trainable_b_v.at(0)) {		
			weight_s.push_back(1);
			weight_s.push_back(num_outs);
		}
		return weight_s;
	}
	unsigned int w_rows = num_inps;
	unsigned int w_cols;
    unsigned int b_rows = 1;
    unsigned int b_cols;
	for (unsigned int i = 0; i < layer_sizes.size(); ++i) {
		w_cols = layer_sizes.at(i);
		b_cols = w_cols;
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
	    b_cols = w_cols;
	    weight_s.push_back(b_rows);
	    weight_s.push_back(b_cols);
	}
	return weight_s;	
}

std::vector<unsigned int> calc_num_weights_layers(const std::vector<unsigned int> & weight_s) { 
	std::vector<unsigned int> n_layer_weights;
    for (unsigned int i = 0; i < weight_s.size() / 4; ++i) {
    	//first product is weight matrix size, second is bias vector size 
        n_layer_weights.push_back(weight_s.at(4 * i) * weight_s.at(4 * i + 1) + weight_s.at(4 * i + 2) * weight_s.at(4 * i + 3));        
    }
    return n_layer_weights;
}

//weight_s incorporates that layers are missing, so can just original usual function
std::vector<unsigned int> calc_num_weights_layers(const std::vector<unsigned int> & weight_s, const std::vector<bool> & trainable_v) {
	return calc_num_weights_layers(weight_s);	
}

std::vector<unsigned int> calc_num_weights_layers(const std::vector<unsigned int> & weight_s, const std::vector<bool> & trainable_w_v, const std::vector<bool> & trainable_b_v) {
	std::vector<unsigned int> n_layer_weights;
	unsigned int j = 0;
    for (unsigned int i = 0; i < trainable_w_v.size(); ++i) {
    	if ((trainable_w_v.at(i)) && (trainable_b_v.at(i))) {
    		n_layer_weights.push_back(weight_s.at(j) * weight_s.at(j + 1) + weight_s.at(j + 2) * weight_s.at(j + 3));
    		j += 4;
    	}
    	else if ((trainable_w_v.at(i)) || (trainable_b_v.at(i))) {
    		n_layer_weights.push_back(weight_s.at(j) * weight_s.at(j + 1));
    		j += 2;	
    	}
    }
    return n_layer_weights;
}

std::vector<unsigned int> get_degen_dependence_lengths(const std::vector<unsigned int> & weight_shapes, const bool & independent) {
	if (independent) {
		std::vector<unsigned int> dependence_lengths = {1};
		return dependence_lengths;
	}
	else {
		unsigned int dependence_length;
		unsigned int num_rows;
		//could reserve but would need to count number of sets of dependent variables 
		//(sum of row numbers in weight_shapes)
		std::vector<unsigned int> dependence_lengths;
		for (unsigned int i = 0; i < weight_shapes.size() / 2; ++i) {
			num_rows = weight_shapes.at(2 * i);
			dependence_length = weight_shapes.at((2 * i) + 1);
			for (unsigned int j = 0; j < num_rows; ++j) {
				dependence_lengths.push_back(dependence_length);
			}
		}
		return dependence_lengths;
	}
}

//see docstring for python implementation of function in python_models/tools.py for explanation of this function
std::vector<unsigned int> get_degen_dependence_lengths2(const std::vector<unsigned int> & weight_shapes, const bool & independent) {
	if (independent) {
		std::vector<unsigned int> dependence_lengths = {1};
		return dependence_lengths;
	}
	else {
		unsigned int dependence_length;
		unsigned int num_rows;
		//could reserve but would need to count number of sets of dependent variables 
		//(sum of row numbers in weight_shapes)
		std::vector<unsigned int> dependence_lengths;
		for (unsigned int i = 0; i < weight_shapes.size() / 2; ++i) {
			num_rows = weight_shapes.at(2 * i);
			dependence_length = weight_shapes.at((2 * i) + 1);
			if (num_rows == 1) { //bias
				for (unsigned int k = 0; k < dependence_length; ++k) {
					dependence_lengths.push_back(1);
				}
			}
			else {
				dependence_lengths.push_back(dependence_length);
				for (unsigned int j = 1; j < num_rows; ++j) {
					for (unsigned int k = 0; k < dependence_length; ++k) {
						dependence_lengths.push_back(1);
					}
				}
			}
		}
		return dependence_lengths;
	}
}

//see equivalent python implementation in python_models/tools.py for explanation of function
std::vector<unsigned int> get_degen_dependence_lengths3(const std::vector<unsigned int> & weight_shapes, const bool & independent) {
	if (independent) {
		std::vector<unsigned int> dependence_lengths = {1};
		return dependence_lengths;
	}
	else {
		unsigned int dependence_length;
		unsigned int num_rows;
		//could reserve but would need to count number of sets of dependent variables 
		//(sum of row numbers in weight_shapes)
		std::vector<unsigned int> dependence_lengths;
		for (unsigned int i = 0; i < weight_shapes.size() / 2 - 2; ++i) {
			num_rows = weight_shapes.at(2 * i);
			dependence_length = weight_shapes.at((2 * i) + 1);
			for (unsigned int j = 0; j < num_rows; ++j) {
				dependence_lengths.push_back(dependence_length);
			}
		}
		num_rows = weight_shapes.at(weight_shapes.size() - 4);
		dependence_length = weight_shapes.at(weight_shapes.size() - 3);
		for (unsigned int i = 0; i < num_rows + 1; ++i) {
			for (unsigned int j = 0; j < dependence_length; ++j) {
				dependence_lengths.push_back(1);
			}
		}
		return dependence_lengths;
	}
}

//see equivalent python implementation in python_models/tools.py for explanation of function
std::vector<unsigned int> get_degen_dependence_lengths4(const std::vector<unsigned int> & weight_shapes, const bool & independent) {
	if (independent) {
		std::vector<unsigned int> dependence_lengths = {1};
		return dependence_lengths;
	}
	else {
		unsigned int dependence_length;
		unsigned int num_rows;
		//could reserve but would need to count number of sets of dependent variables 
		//(sum of row numbers in weight_shapes)
		std::vector<unsigned int> dependence_lengths;
		for (unsigned int i = 0; i < weight_shapes.size() / 2 - 2; ++i) {
			num_rows = weight_shapes.at(2 * i);
			dependence_length = weight_shapes.at((2 * i) + 1);
			if (num_rows == 1) { //bias
				for (unsigned int k = 0; k < dependence_length; ++k) {
					dependence_lengths.push_back(1);
				}
			}
			else {
				dependence_lengths.push_back(dependence_length);
				for (unsigned int j = 1; j < num_rows; ++j) {
					for (unsigned int k = 0; k < dependence_length; ++k) {
						dependence_lengths.push_back(1);
					}
				}
			}
		}
		num_rows = weight_shapes.at(weight_shapes.size() - 4);
		dependence_length = weight_shapes.at(weight_shapes.size() - 3);
		for (unsigned int i = 0; i < num_rows + 1; ++i) {
			for (unsigned int j = 0; j < dependence_length; ++j) {
				dependence_lengths.push_back(1);
			}
		}
		return dependence_lengths;
	}
}

std::vector<unsigned int> get_hyper_dependence_lengths(const std::vector<unsigned int> & weight_shapes, const std::string & granularity) {
	if (granularity == "single") {
		std::vector<unsigned int> hyper_dependence_lengths = {1};
		return hyper_dependence_lengths;
	}
	else if (granularity == "layer") {
		return calc_num_weights_layers(weight_shapes);
	}
	else if (granularity == "input_size") {
		bool indp = false;
		return get_degen_dependence_lengths(weight_shapes, indp);
	}
	else { //this is just to get rid of warning about not returning anything in case of other conditionals not being satisfied
		std::cout << "granularity not valid. returning null vector" << std::endl;
		std::vector<unsigned int> hyper_dependence_lengths;
		return hyper_dependence_lengths;
	}
}

//could definitely template these instead
//-------------------------------------------------------------------
//creates a vector comprised of the seq_values[i] values, each of which is included (consecutively) in the vector seq_lengths[i] times.
// e.g. get_seq_vec({0.1, 0.9, 0.1, 0.9}, {1,2,3,4}) = {0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9}
std::vector<double> get_seq_vec(std::vector<double> seq_values, std::vector<unsigned int> seq_lengths) {
	std::vector<double> seq;
	//could sum seq_values and reserve space here but probably not worth effort
	for (unsigned int i = 0; i < seq_lengths.size(); ++i) {
		unsigned int seq_length = seq_lengths.at(i);
		for (unsigned int j = 0; j < seq_length; ++j) {
			seq.push_back(seq_values.at(i));
		}
	}
	return seq;
}

std::vector<unsigned int> get_seq_vec(std::vector<unsigned int> seq_values, std::vector<unsigned int> seq_lengths) {
	std::vector<unsigned int> seq;
	for (unsigned int i = 0; i < seq_lengths.size(); ++i) {
		unsigned int seq_length = seq_lengths.at(i);
		for (unsigned int j = 0; j < seq_length; ++j) {
			seq.push_back(seq_values.at(i));
		}
	}
	return seq;
}
//-------------------------------------------------------------------