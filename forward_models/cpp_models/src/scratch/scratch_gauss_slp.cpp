/* include external code */
#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <fstream>
#include <typeinfo>

/* include in-house code */
#include "scratch/scratch_gauss_slp.hpp"

double relu(double x) {
    if (x > 0.) {
        return x;
    }
    else { 
        return 0.;
    }
}

//functor for testing unary_expr method in eigen
double relu_c::operator()(double x) { 
    if (x > 0.) {
        return x;
    }
    else {
        return 0.;    
    }        
}

//should be inlined
double const_pi() { 
    return std::atan(1)*4; 
}

forward_prop::forward_prop(uint num_inputs_, uint num_outputs_, uint m_, uint batch_size_, std::vector<uint> layer_sizes_, std::string x_path_, std::string y_path_) : 
    //it seems that even if initialiser isn't included implicitly, it is called implicitly, and default constructor for each object is called
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
    //graveyard from endless debugging of eigen map seg fault
    //x_tr_m << 0,1,2,3,4,5,6,7,8,9,10,11;
    //y_tr_m << 13,36,71,118,177,248,331,426,533,652,783,926;
    // x_tr_m(x_tr_v.data(), m, num_inputs), y_tr_m(y_tr_v.data(), m, num_outputs)
    //note this is "placement new", not "normal" new
    // new (&x_tr_m) Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(x_tr_v.data(), m, num_inputs); 
    // y_tr_v = get_tr_vec(num_outputs, y_path);
    // new (&y_tr_m) Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(y_tr_v.data(), m, num_outputs);
}

std::vector<double> forward_prop::get_tr_vec(const uint & num_io, const std::string & path) { 
    std::cout << "got to beginning of get tr vec" << std::endl;
    const uint num_data = calc_num_data(num_io);
    std::vector<double> tr_v = file_2_vec(path, num_data);
    std::cout << "got to end of get tr vec" << std::endl;
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
    values.reserve(num_data); 
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
    }
    return values;
}

std::vector<uint> forward_prop::get_weight_shapes() { //return is stored at location of vector which calls function in its initialisation
    std::vector<uint> weight_s;
    weight_s.reserve((layer_sizes.size() + 1) * 2); 
    uint w_rows = num_inputs;
    weight_s.push_back(w_rows);
    uint w_cols = layer_sizes.front();
    weight_s.push_back(w_cols);
    uint b_rows = w_cols; 
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
        std::random_shuffle(rand_m_ind.data(), rand_m_ind.data() + m); 
        x_tr_b = x_tr_m(rand_m_ind.segment(0, batch_size), Eigen::placeholders::all);
        y_tr_b = y_tr_m(rand_m_ind.segment(0, batch_size), Eigen::placeholders::all);
        b_c = 2;        
    }
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > batch_v = {x_tr_b, y_tr_b};
    return batch_v;
}

//change this to setup_ll method eventually.
//will set function pointer for likelihood accordingly so don't need if conditional in ()operator
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

//should be able to make these functions rather than methods. for non-batch ones, will have to reference x
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

//could also make these functions (and pass reference to y for non-batch), but didn't in python versions
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

//version of scratch_gauss_slp_func which uses static variables for all parameters which don't need to be re-calculated for each llhood call. could probably be used with polychord.
//for comments on function(s) implementation see scratch_gauss_slp_func and the called functions
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
double scratch_gauss_slp_static_ll(Eigen::VectorXd & w) {  
    //in class implementation statics will be declared and defined in constructor and other initialisation
    //methods
    //------------------------------------------------------------------------------------------------------------
    static const uint num_inputs = 2;
    static const uint num_outputs = 2;
    static const uint m = 6;
    static const uint batch_size = 4;
    static const std::vector<uint> layer_sizes = std::vector<uint>{5};
    static const std::string x_path = "./data/scratch_gauss_slp_x.txt";
    static const std::string y_path = "./data/scratch_gauss_slp_y.txt";
    static const double LL_var = 1.;
    static const std::string LL_type = "gauss";
    static const std::vector<uint> weight_shapes = get_weight_shapes(num_inputs, layer_sizes, num_outputs);
    static std::vector<double> x_tr_v = get_tr_vec(m, num_inputs, x_path);
    static Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x_tr_m(x_tr_v.data(), m, num_inputs);
    static std::vector<double> y_tr_v = get_tr_vec(m, num_outputs, y_path);
    static Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y_tr_m(y_tr_v.data(), m, num_outputs);
    static const double LL_norm = calc_LL_norm(LL_var, m, num_outputs, LL_type);
    static const uint num_complete_batches = m / batch_size;
    static const uint num_batches = static_cast<uint>(ceil(static_cast<double>(m) / batch_size));
    //------------------------------------------------------------------------------------------------------------
    //following are called each time polychord needs to evaluate likelihood
    //i.e the ()operator should call these two functions, and take w as its argument
    //get batches stuff is in the get_ll function
    std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > weight_matrices = get_weight_matrices(w, weight_shapes);
    double LL = get_LL(x_tr_m, y_tr_m, weight_matrices, m, batch_size, num_complete_batches, num_batches, LL_var, LL_norm, LL_type);
    return LL;
    //------------------------------------------------------------------------------------------------------------

}
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------


//test function which calls functions to calculate ll
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
void scratch_gauss_slp_func(uint num_inputs_, uint num_outputs_, uint m_, uint batch_size_, std::vector<uint> layer_sizes_, std::string x_path_, std::string y_path_) {  
    const uint num_inputs = num_inputs_;
    const uint num_outputs = num_outputs_;
    const uint m = m_;
    const uint batch_size = batch_size_;
    const std::vector<uint> layer_sizes = layer_sizes_;
    const std::string x_path = x_path_;
    const std::string y_path = y_path_;
    const double LL_var = 1.;
    const std::string LL_type = "gauss";
    const std::vector<uint> weight_shapes = get_weight_shapes(num_inputs, layer_sizes, num_outputs);

    // nrvo doesn't seem to work for eigen maps, so data inside std::vector is destroyed upon the function exiting
    // leaving a dangling pointer
    // Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x_tr_m = get_tr_map(m, num_inputs, x_path); 
    //following also leads to dangling pointer. sort of makes sense nrvo on std::vector can't be done here.
    // Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x_tr_m(get_tr_vec(m, num_inputs, x_path).data(), m, num_inputs);
    std::vector<double> x_tr_v = get_tr_vec(m, num_inputs, x_path);
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x_tr_m(x_tr_v.data(), m, num_inputs);
    std::vector<double> y_tr_v = get_tr_vec(m, num_outputs, y_path);
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y_tr_m(y_tr_v.data(), m, num_outputs);
    const double LL_norm = calc_LL_norm(LL_var, m, num_outputs, LL_type);
    //possibly could use boolean for whether remainder batch is needed, but can't be bothered as would require modifying
    //get_batches() slightly
    const uint num_complete_batches = m / batch_size;
    const uint num_batches = static_cast<uint>(ceil(static_cast<double>(m) / batch_size));
    // generate weights manually for test. not needed in production
    //---------------------------------------------------------------------------
    const uint num_weights = calc_num_weights(num_inputs, layer_sizes, num_outputs);
    Eigen::VectorXd w = Eigen::VectorXd::LinSpaced(num_weights, 0, num_weights - 1);
    //---------------------------------------------------------------------------
    std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > weight_matrices = get_weight_matrices(w, weight_shapes);
    double LL = get_LL(x_tr_m, y_tr_m, weight_matrices, m, batch_size, num_complete_batches, num_batches, LL_var, LL_norm, LL_type);
}

//eigen maps don't appear to work for nrvo. thus have to do nrvo on std::vector, then use map on it
//Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > get_tr_map(const uint & m, const uint & num_io, const std::string & path) {
std::vector<double> get_tr_vec(const uint & m, const uint & num_io, const std::string & path) { 
    const uint num_data = calc_num_data(m, num_io);
    std::vector<double> tr_v = file_2_vec(path, num_data);
    return tr_v;
}

uint calc_num_data(const uint & m, const uint & num_io){
    return m * num_io;
}

//depracated
uint calc_num_x(const uint & m, const uint & num_inps) {
    return m * num_inps;
}

//depracated
uint calc_num_y(const uint & m, const uint & num_outs) {
    return m * num_outs;
}

std::vector<double> file_2_vec(const std::string & path, const uint & num_data) {
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

std::vector<uint> get_weight_shapes(const uint & num_inps, const std::vector<uint> & layer_sizes, const uint & num_outs) {
    //this probably isn't 100% efficient or concise code and there are some redundant
    // statements but I can't be bothered since it's a small function
    std::vector<uint> weight_shapes;
    weight_shapes.reserve((layer_sizes.size() + 1) * 2); //+1 for output layer, *2 for biases
    uint w_rows = num_inps;
    weight_shapes.push_back(w_rows);
    uint w_cols = layer_sizes.front();
    weight_shapes.push_back(w_cols);
    uint b_rows = w_cols; //should always be same as w_cols, but perhaps not for complex nns
    weight_shapes.push_back(b_rows);
    uint b_cols = 1;
    weight_shapes.push_back(b_cols);
    for (uint i = 1; i < layer_sizes.size(); ++i) {
        w_rows = w_cols;
        weight_shapes.push_back(w_rows);
        w_cols = layer_sizes[i];
        weight_shapes.push_back(w_cols);
        b_rows = w_cols;
        weight_shapes.push_back(b_rows);
        weight_shapes.push_back(b_cols);        
    }
    w_rows = w_cols;
    weight_shapes.push_back(w_rows);
    w_cols = num_outs;
    weight_shapes.push_back(w_cols);
    b_rows = w_cols;
    weight_shapes.push_back(b_rows);
    weight_shapes.push_back(b_cols);
    return weight_shapes;
}

//only needed during testing, when manually generating weight std::vector
uint calc_num_weights(const uint & num_inps, const std::vector<uint> & layer_sizes, const uint & num_outs) {
    //ditto
    uint n = (num_inps + 1) * layer_sizes.front();
    for (uint i = 1; i < layer_sizes.size(); ++i) {
        n += (layer_sizes[i-1] + 1) * layer_sizes[i];
    }
    n += (layer_sizes.back() + 1) * num_outs;
    return n;
}

//first argument may be map if polychord works with vectors or vanilla arrays/pointers.
//if it works with eigen objects nothing needs to be changed here
//note this actually returns vector of maps (which point to original weight vector), not matrices.
//but maps act like matrices (they're a pointer which enables the underlying to be manipulated as an eigen matrix)
//the fact that the (1d) biases are mapped to matrices is problematic for the nn calculation functions.
//however, as far as i know std::vectors have to be homogeneous, so can't have eigen and vector maps in the same 
// vector
//perhaps if i can make a std::vector of some class which both matrices and vectors (or their maps) inherit from,
//i could store matrix and vector maps in the same std::vector (probably the optimal solution).
//if not could divide into two std::vectors, one to hold multiplicative weights as matrices, and one
//to hold bias terms as eigen vectors. don't think i could return both of these from same function
//(though i could pass them to it by reference and edit)
//alternatively could have two separate functions for biases and multip weights, but inefficient
std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > get_weight_matrices(Eigen::VectorXd & w, const std::vector<uint> & weight_shapes){
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

std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> get_batches(Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & x_tr_m, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y_tr_m, const uint & m, const uint & batch_size, const uint & num_complete_batches, const uint & num_batches) {
    static uint b_c = 1;
    static Eigen::Matrix<uint, Eigen::Dynamic, 1> rand_m_ind = Eigen::Matrix<uint, Eigen::Dynamic, 1>::LinSpaced(m, 0, m - 1); //could pass these static variables in by references instead
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

double calc_LL_norm(const double & LL_var, const uint & m, const uint & num_outputs, const std::string & LL_type) {
    if (LL_type == "gauss") {
        const uint LL_dim = m * num_outputs;
        return -0.5 * LL_dim * (std::log(2. * const_pi()) + std::log(LL_var));
    }
    else if (LL_type == "categorical_crossentropy") {
        return 0.;
    }
    else {
        std::cout<< "other llhoods not implemented yet" << std::endl;
        return EXIT_FAILURE;
    }
}

//pass function pointers pointing at functions for nn calculation so nn architecture can be determined at initialisation 
//pass function pointers pointing at ll function, which is determined at initialisation so don't need conditional over ll type
//could also pass function pointer (determined at initialisation) which depends on whether using batches or not.
//if using full batch, proceed to calculate nn from full tr map. if not, modify get_batches function so it returns maps to matrices (lose a bit of efficiency here), then maps from either full or partial batch can be used to calc nn without conditional over batch_size. in this case, templates for nn and ll no longer needed, as maps always used for tr data.
//an alternative to modifying get_batches to return a map instead of a matrix is to having the function pointer being type map for the full tr, and matrix for batches. then we initialising x and y variables to be used for nn and ll calcs, will have to use auto keyword to derive their types from the function pointer return type.
double get_LL(Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & x_tr_m, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y_tr_m, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w, const uint & m, const uint & batch_size, const uint & num_complete_batches, const uint & num_batches, const double & LL_var, const double & LL_norm, const std::string & LL_type) {
    double LL;
    if (batch_size == m) {
        Eigen::MatrixXd pred = slp_nn(x_tr_m, w);
        if (LL_type == "gauss") {
            LL = calc_gauss_ll(LL_var, LL_norm, y_tr_m, pred);
        }
        else if(LL_type == "categorical_crossentropy") {
            LL = calc_ce_ll(LL_var, LL_norm, y_tr_m, pred);
        }
    }
    else if (batch_size < m) {
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> batch_m = get_batches(x_tr_m, y_tr_m, m, batch_size, num_complete_batches, num_batches);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x_tr_b = batch_m[0];
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y_tr_b = batch_m[1];
        Eigen::MatrixXd pred = slp_nn_batch(x_tr_b, w);
                if (LL_type == "gauss") {
            LL = calc_gauss_ll_batch(LL_var, LL_norm, y_tr_b, pred);
        }
        else if(LL_type == "categorical_crossentropy") {
            LL = calc_ce_ll_batch(LL_var, LL_norm, y_tr_b, pred);
        }
    }
    std::cout << LL << std::endl;
    return LL;
}
//following should be turned into templates which accept either a map or an actual matrix for x for nn calculating functions (y for llhood calcs), as operations on both are identical
//------------------------------------------------------------------------------------------------------------------ 

//currently mapping bias matrices to eigen vectors, as .rowwise() operation requires a vector not a 1xn matrix.
//this is annoying as we have to get size of matrix to do this mapping.
//could alternatively use .replicate on 1xn matrix to make it same size as a * w and just add them together, but this requires knowing first dimension of a * w. for full batch this isn't problem, as it's always going to be m.
//for mini batches, will always be batch_size, apart from if there's a remainder, so just in case, need to calculate it every time...
//life would be much easier if biases came in as vectors... (see get_weight_matrices() comments)
//perhaps there is another way to add matrix of size 1xn to a * w, but i don't know one

Eigen::MatrixXd slp_nn(Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose()).unaryExpr(std::ptr_fun(relu));
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]) + w[1].transpose().replicate(m, 1)).unaryExpr(std::ptr_fun(relu)); //have to pass in m
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + w[1].transpose()).unaryExpr(std::ptr_fun(relu)); //only works if w[1] is an eigen vector
    Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[3].data(), w[3].size()).transpose());
    //Eigen::MatrixXd a2 = (a1 * w[2]) + w[3].transpose().replicate(a1.rows(), 1);
    //Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + w[3].transpose();
    return a2;
}

Eigen::MatrixXd slp_nn_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose()).unaryExpr(std::ptr_fun(relu));
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]) + w[1].transpose().replicate(x.rows(), 1)).unaryExpr(std::ptr_fun(relu)); //can .replicate() batch_size number of times, unless there's a remainder...
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + w[1].transpose()).unaryExpr(std::ptr_fun(relu));
    Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[3].data(), w[3].size()).transpose());
    //Eigen::MatrixXd a2 = (a1 * w[2]) + w[3].transpose().replicate(a1.rows(), 1);
    //Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + w[3].transpose();
    return a2;
}

double calc_gauss_ll(const double & LL_var, const double & LL_norm, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y, Eigen::MatrixXd & pred) {
    const double chi_sq = -1. / (2. * LL_var) * (pred - y).squaredNorm();
    return chi_sq + LL_norm;
}

double calc_gauss_ll_batch(const double & LL_var, const double & LL_norm, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred) {
    const double chi_sq = -1. / (2. * LL_var) * (pred - y).squaredNorm();
    return chi_sq + LL_norm;
}

double calc_ce_ll(const double & LL_var, const double & LL_norm, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y, Eigen::MatrixXd & pred) {
    return -1. * (pred.array().log() * y.array()).sum();
}

double calc_ce_ll_batch(const double & LL_var, const double & LL_norm, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred) {
    return -1. * (pred.array().log() * y.array()).sum();
}


//------------------------------------------------------------------------------------------------------------------ 

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------

// single function which calculates ll
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
void scratch_gauss_slp_full() {
    //initialise stuff
    const int num_inputs = 2;
    const int num_outputs = 2;
    const int m = 6;
    const int batch_size = 3;
    const int a1_size = 5;
    const int num_weights = (num_inputs + 1) * a1_size + (a1_size + 1) * num_outputs;
    //call data read in init
    //read in data from csv. one time call at beginning
    //call separately for x and y
    const unsigned int num_x_data = m * num_inputs;
    const unsigned int num_y_data = m * num_outputs; //assumes output is one-hot encoded vector in case of classificiation
    //const unsigned int num_data = num_x_data + num_y_data;
    const std::string x_path = "./data/scratch_gauss_slp_x.txt";
    std::ifstream x_indata;
    x_indata.open(x_path);
    std::string x_line;
    std::vector<double> x_values;
    x_values.reserve(num_x_data); //saves having to reallocate many, many times...
    while (std::getline(x_indata, x_line)) {
        std::stringstream lineStream(x_line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            x_values.push_back(std::stod(cell));
        }
    }
    const std::string y_path = "./data/scratch_gauss_slp_y.txt";
    std::ifstream y_indata;
    y_indata.open(y_path);
    std::string y_line;
    std::vector<double> y_values;
    y_values.reserve(num_y_data); //saves having to reallocate many, many times...
    while (std::getline(y_indata, y_line)) {
        std::stringstream lineStream(y_line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            y_values.push_back(std::stod(cell));
        }
    }    
    //create eigen map objects which point to training data. one time operations at beginning of program
    //could have used one file and one map to get x_tr and y_tr using custom strides. however, x_tr, y_tr wouldn't be stored in contigous memory so might have slowed matrix operations down during execution. could have laid out data in file differently e.g. x_tr rows then y_tr rows, but might be a pain saving this format if num_inputs != num_outputs
    //Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > tot_tr_m(values.data(), m, num_inputs + num_outputs);
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x_tr_m(x_values.data(), m, num_inputs); //rowmajor faster for x in x * w I think
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y_tr_m(y_values.data(), m, num_outputs);

    //mini-batch stuff. note first set of mini-batches aren't randomly shuffled
    Eigen::VectorXi rand_m_ind = Eigen::VectorXi::LinSpaced(m, 0, m - 1);
    const unsigned int num_complete_batches = m / batch_size;
    const unsigned int num_batches = static_cast<unsigned int>(ceil(static_cast<double>(m) / batch_size));
    unsigned int b_c = 1;
    unsigned int b_start;
    Eigen::MatrixXd x_tr_b;
    Eigen::MatrixXd y_tr_b;
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
    //alternatively, might be possible to get subset of training map using same indexing as above, but since memory won't be contiguous, probably inefficient. might be better alternative if batch_size large
    // another alternative is to save the original training data as a matrix and release vector memory (create new matrix using Eigen::MatrixXd m(std_vec.data())), permute it in-place, then at each iteration create a map for each batch. also might be better if batch_size is large 
    // a slight alteration to previous method is to make matrix of training from map, then proceed as before (slower, but less changing methodology)
    //not sure which of these two is faster
    //Eigen::MatrixXd x_tr_b = x_tr_m(rand_m_ind, Eigen::placeholders::all);
    //Eigen::MatrixXd x_tr_b = indices.asPermutation() * x_tr_m;  

    //nn computation
    // generate weights manually for test. not needed in production
    Eigen::VectorXd w = Eigen::VectorXd::LinSpaced(num_weights, 0, num_weights - 1);
    unsigned int start_index = 0;
    const unsigned int w1_len = num_inputs * a1_size; // in real implementation this will be uninitialised vector
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w1(w.segment(start_index, w1_len).data(), num_inputs, a1_size); //rowmajor to match numpy. probably less efficient but better for consistency
    //following would fill weight matrix in different order to python versions (which are rowmajor)
    //Eigen::Map<Eigen::MatrixXd> w1(w.segment(start_index, w1_len).data(), num_inputs, a1_size); //colmajor faster for matrix w in x * w I think
    start_index += w1_len;
    const unsigned int b1_len = a1_size;
    Eigen::Map<Eigen::VectorXd> b1(w.segment(start_index, b1_len).data(), b1_len);
    start_index += b1_len;
    const unsigned int w2_len = a1_size * num_outputs;
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w2(w.segment(start_index, w2_len).data(), a1_size, num_outputs); 
    //Eigen::Map<Eigen::MatrixXd> w2(w.segment(start_index, w2_len).data(), a1_size,  num_outputs);
    start_index += w2_len;
    const unsigned int b2_len = num_outputs;
    Eigen::Map<Eigen::VectorXd> b2(w.segment(start_index, b2_len).data(), b2_len);
    //ptr_fun depracated, try std::function or std::red. also try functor and/or function pointer
    //function, pointer to function, functor of function (!?) don't work. std::ptr_fun(function) does work, but depracated from c++11
    Eigen::MatrixXd a1 = ((x_tr_m * w1).rowwise() + b1.transpose()).unaryExpr(std::ptr_fun(relu)); //could be beneficial to make activations rowwise
    Eigen::MatrixXd a2 = ((a1 * w2).rowwise() + b2.transpose());

    //ll computation
    const double LL_var = 1.;
    const double chi_sq = -1. / (2. * LL_var) * (a2 - y_tr_m).squaredNorm();
    const unsigned int LL_dim = m * num_outputs;
    const double LL_norm = -0.5 * LL_dim * (std::log(2. * const_pi()) + std::log(LL_var));
    const double LL = chi_sq + LL_norm;
    std::cout << LL << std::endl;

}

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------