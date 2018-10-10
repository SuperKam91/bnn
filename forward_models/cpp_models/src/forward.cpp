/* external codebase */
#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <fstream>

/* in-house code */
#include "forward.hpp"

double relu(double x) {
    if (x > 0.) {
        return x;
    }
    else { 
        return 0.;
    }
}

double const_pi() { 
    return std::atan(1)*4; 
}

forward_prop::forward_prop(uint num_inputs_, uint num_outputs_, uint m_, uint batch_size_, std::vector<uint> layer_sizes_, std::string x_path_, std::string y_path_) : 
	num_inputs(num_inputs_), 
	num_outputs(num_outputs_), 
	m(m_), 
	batch_size(batch_size_), 
	layer_sizes(layer_sizes_), 
	x_path(x_path_), 
	y_path(y_path_), 
	weight_shapes(get_weight_shapes()), 
	x_tr_v(get_tr_vec(num_inputs, x_path)), 
	x_tr_m(x_tr_v.data(), m, num_inputs), 
	y_tr_v(get_tr_vec(num_outputs, y_path)), 
	y_tr_m(y_tr_v.data(), m, num_outputs), 
	LL_type(), 
	LL_var(1.), 
	LL_norm(0.), 
	num_complete_batches(m / batch_size), 
	num_batches(static_cast<uint>(ceil(static_cast<double>(m) / batch_size))), 
	b_c(1.), 
	rand_m_ind(Eigen::Matrix<uint, Eigen::Dynamic, 1>::LinSpaced(m, 0, m - 1)) {
    //x_tr_m << 0,1,2,3,4,5,6,7,8,9,10,11;
    //y_tr_m << 13,36,71,118,177,248,331,426,533,652,783,926;
    // x_tr_m(x_tr_v.data(), m, num_inputs), y_tr_m(y_tr_v.data(), m, num_outputs)
    //get_tr_vec(num_inputs, x_path)
    //get_tr_vec(num_outputs, y_path)
    // LL_var = 1.; //as with python implementations, in future can take LL_var as arg in constructor or ll setup
    // x_tr_v = get_tr_vec(num_inputs, x_path);
    //note this is "placement new", not "normal" new
    // new (&x_tr_m) Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(x_tr_v.data(), m, num_inputs); 
    // y_tr_v = get_tr_vec(num_outputs, y_path);
    // new (&y_tr_m) Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(y_tr_v.data(), m, num_outputs);
}

std::vector<double> forward_prop::get_tr_vec(const uint & num_io, const std::string & path) { 
    const uint num_data = calc_num_data(num_io);
    std::vector<double> tr_v = file_2_vec(path, num_data);
    return tr_v;
}

uint forward_prop::calc_num_data(const uint & num_io){
    return m * num_io;
}

std::vector<double> forward_prop::file_2_vec(const std::string & path, const uint & num_data) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    values.reserve(num_data); //saves having to reallocate many, many times...
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
    }
    return values;
}

std::vector<uint> forward_prop::get_weight_shapes() { //this is using nrvo - GOOD
	std::vector<uint> weight_s;
    weight_s.reserve((layer_sizes.size() + 1) * 2); //+1 for output layer, *2 for biases
    uint w_rows = num_inputs;
    weight_s.push_back(w_rows);
    uint w_cols = layer_sizes.front();
    weight_s.push_back(w_cols);
    uint b_rows = w_cols; //should always be same as w_cols, but perhaps not for complex nns
    weight_s.push_back(b_rows);
    uint b_cols = 1;
    weight_s.push_back(b_cols);
    for (uint i = 1; i < layer_sizes.size(); ++i) {
        w_rows = w_cols;
        weight_s.push_back(w_rows);
        w_cols = layer_sizes[i];
        weight_s.push_back(w_cols);
        b_rows = w_cols;
        weight_s.push_back(b_rows);
        weight_s.push_back(b_cols);        
    }
    w_rows = w_cols;
    weight_s.push_back(w_rows);
    w_cols = num_outputs;
    weight_s.push_back(w_cols);
    b_rows = w_cols;
    weight_s.push_back(b_rows);
    weight_s.push_back(b_cols);
    return weight_s;
}

std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > forward_prop::get_weight_matrices(Eigen::VectorXd & w){
    uint start_index = 0;
    //assumes all matrices are 2d
    const unsigned long int num_matrices = weight_shapes.size() / 2; //.size() returns long uint, so make num_matrices this to get rid of warning
    std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > weight_matrices;
    weight_matrices.reserve(num_matrices);
    uint weight_start = 0;
    uint weight_length;
    for (uint i = 0; i < num_matrices; ++i) {
        weight_length = weight_shapes[2 * i] * weight_shapes[2 * i + 1];
        weight_matrices.push_back(Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > (w.segment(weight_start, weight_length).data(), weight_shapes[2 * i], weight_shapes[2 * i + 1])); //rowmajor to match python. probably less efficient but better for consistency
        weight_start += weight_length;
    }
    return weight_matrices;
}

std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> forward_prop::get_batches() {
    uint b_start;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_tr_b;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> y_tr_b;
    b_start = (b_c - 1) * batch_size; 
    if (b_c <= num_complete_batches){
        x_tr_b = x_tr_m(rand_m_ind.segment(b_start, batch_size), Eigen::placeholders::all);
        y_tr_b = y_tr_m(rand_m_ind.segment(b_start, batch_size), Eigen::placeholders::all);
        ++b_c;
    }
    else if (b_c == num_complete_batches + 1 && num_batches > num_complete_batches) {
        x_tr_b = x_tr_m(rand_m_ind.segment(b_start, m - b_start), Eigen::placeholders::all);
        y_tr_b = y_tr_m(rand_m_ind.segment(b_start, m - b_start), Eigen::placeholders::all);
        ++b_c;        
    }
    else {
        std::random_shuffle(rand_m_ind.data(), rand_m_ind.data() + m); //re-shuffle data
        x_tr_b = x_tr_m(rand_m_ind.segment(0, batch_size), Eigen::placeholders::all);
        y_tr_b = y_tr_m(rand_m_ind.segment(0, batch_size), Eigen::placeholders::all);
        b_c = 2;        
    }
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > batch_v = {x_tr_b, y_tr_b};
    return batch_v;
}

//change this to setup_ll method eventually
void forward_prop::calc_LL_norm(std::string LL_type_) {
    LL_type = LL_type_;
    if (LL_type == "gauss") {
        const uint LL_dim = m * num_outputs;
        LL_norm = -0.5 * LL_dim * (std::log(2. * const_pi()) + std::log(LL_var));
    }
    else if (LL_type == "categorical_crossentropy") {
        LL_norm = 0.;
    }
    else {
        std::cout<< "other llhoods not implemented yet. please quit program." << std::endl;
        LL_norm = EXIT_FAILURE;
    }
}

double forward_prop::operator()(Eigen::VectorXd & w) {
    double LL;
    std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > weight_matrices = get_weight_matrices(w);
    if (batch_size == m) {
        Eigen::MatrixXd pred = slp_nn(weight_matrices);
        if (LL_type == "gauss") {
            LL = calc_gauss_ll(pred);
        }
        else if(LL_type == "categorical_crossentropy") {
            LL = calc_ce_ll(pred);
        }
    }
    else if (batch_size < m) {
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> batch_m = get_batches();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x_tr_b = batch_m[0];
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y_tr_b = batch_m[1];
        Eigen::MatrixXd pred = slp_nn_batch(x_tr_b, weight_matrices);
            if (LL_type == "gauss") {
                LL = calc_gauss_ll_batch(y_tr_b, pred);
        }
            else if(LL_type == "categorical_crossentropy") {
                LL = calc_ce_ll_batch(y_tr_b, pred);
        }
    }
    std::cout << LL << std::endl;
    return LL;
}

Eigen::MatrixXd forward_prop::slp_nn(std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x_tr_m * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose()).unaryExpr(std::ptr_fun(relu));
    Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[3].data(), w[3].size()).transpose());
    return a2;
}

Eigen::MatrixXd forward_prop::slp_nn_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose()).unaryExpr(std::ptr_fun(relu));
    Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[3].data(), w[3].size()).transpose());
    return a2;
}

double forward_prop::calc_gauss_ll(Eigen::MatrixXd & pred) {
    const double chi_sq = -1. / (2. * LL_var) * (pred - y_tr_m).squaredNorm();
    return chi_sq + LL_norm;
}

double forward_prop::calc_gauss_ll_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred) {
    const double chi_sq = -1. / (2. * LL_var) * (pred - y).squaredNorm();
    return chi_sq + LL_norm;
}

double forward_prop::calc_ce_ll(Eigen::MatrixXd & pred) {
    return -1. * (pred.array().log() * y_tr_m.array()).sum();
}

double forward_prop::calc_ce_ll_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred) {
    return -1. * (pred.array().log() * y.array()).sum();
}

//only used when weights need to be generated manually
uint calc_num_weights(const uint & num_inps, const std::vector<uint> & layer_sizes, const uint & num_outs) {
    uint n = (num_inps + 1) * layer_sizes.front();
    for (uint i = 1; i < layer_sizes.size(); ++i) {
        n += (layer_sizes[i-1] + 1) * layer_sizes[i];
    }
    n += (layer_sizes.back() + 1) * num_outs;
    return n;
}