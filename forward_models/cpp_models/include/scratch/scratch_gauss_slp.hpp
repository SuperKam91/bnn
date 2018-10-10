//had to give full function input signature for certain functions which had eigen objects as arguments. may be because they take templates as arguments?
#pragma once /* ensures stuff in header only compiled once */
#include <string>
#include <vector>
#include <Eigen/Core>
#include <iostream>

double relu(double);

class relu_c {
public:
    double operator()(double);
};

double const_pi();

class forward_prop {
public:
    forward_prop(uint, uint, uint, uint, std::vector<uint>, std::string, std::string);
    void calc_LL_norm(std::string);
    double operator()(Eigen::VectorXd & w);     
private:
    //member variables
    const uint num_inputs;
    const uint num_outputs;
    const uint m;
    const uint batch_size;
    const std::vector<uint> layer_sizes;
    const std::string x_path;
    const std::string y_path;
    std::vector<uint> weight_shapes;
    std::vector<double> x_tr_v;
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x_tr_m;
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_tr_m;
    std::vector<double> y_tr_v;
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y_tr_m;
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> y_tr_m;
    //can't make these five constant, without assigning their values here
    //---------------------------------------------------------------------------------------------------------------- 
    std::string LL_type; 
    double LL_var;
    double LL_norm; 
    uint num_complete_batches;
    uint num_batches;
    uint b_c;
    //----------------------------------------------------------------------------------------------------------------
    Eigen::Matrix<uint, Eigen::Dynamic, 1> rand_m_ind;
    //member methods
    std::vector<uint> get_weight_shapes();
    std::vector<double> get_tr_vec(const uint &, const std::string &);
    uint calc_num_data(const uint &);
    std::vector<double> file_2_vec(const std::string &, const uint &);
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> get_batches();
    std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > get_weight_matrices(Eigen::VectorXd & w);
    Eigen::MatrixXd slp_nn(std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w);
    Eigen::MatrixXd slp_nn_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w);
    double calc_gauss_ll(Eigen::MatrixXd & pred); //if get rid of batch conditional this will take map y as arg as well
    double calc_gauss_ll_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred);
    double calc_ce_ll(Eigen::MatrixXd & pred); //if get rid of batch conditional this will take map y as arg as well
    double calc_ce_ll_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred);
};

double scratch_gauss_slp_static_ll(Eigen::VectorXd & w);

void scratch_gauss_slp_func(uint, uint, uint, uint, std::vector<uint>, std::string, std::string);

std::vector<uint> get_weight_shapes(const uint &, const std::vector<uint> &, const uint &);

uint calc_num_weights(const uint &, const std::vector<uint> &, const uint &);

//Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > get_tr_map(const uint &, const uint &, const std::string &);
std::vector<double> get_tr_vec(const uint & m, const uint & num_io, const std::string & path);

uint calc_num_data(const uint &, const uint &);

uint calc_num_x(const uint &, const uint &);

uint calc_num_y(const uint &, const uint &);

std::vector<double> file_2_vec(const std::string &, const uint &);

double calc_LL_norm(const double &, const uint &, const uint &, const std::string &);

std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> get_batches(Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & x_tr_m, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y_tr_m, const uint & m, const uint & batch_size, const uint & num_complete_batches, const uint & num_batches); 

std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > get_weight_matrices(Eigen::VectorXd & w, const std::vector<uint> & weight_shapes); 

double get_LL(Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & x_tr_m, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y_tr_m, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w, const uint & m, const uint & batch_size, const uint & num_complete_batches, const uint & num_batches, const double & LL_var, const double & LL_norm, const std::string & LL_type);

Eigen::MatrixXd slp_nn(Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w);

Eigen::MatrixXd slp_nn_batch(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w);

double calc_gauss_ll(const double & LL_var, const double & LL_norm, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y, Eigen::MatrixXd & pred);

double calc_gauss_ll_batch(const double & LL_var, const double & LL_norm, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred);

double calc_ce_ll(const double & LL_var, const double & LL_norm, Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > & y, Eigen::MatrixXd & pred);

double calc_ce_ll_batch(const double & LL_var, const double & LL_norm, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y, Eigen::MatrixXd & pred);

void scratch_gauss_slp_full();
