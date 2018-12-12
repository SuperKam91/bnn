#pragma once

/* external codebases */
#include <string>
#include <vector>
#include <Eigen/Dense>

class forward_prop {
public:
	forward_prop(uint, uint, uint, uint, std::vector<uint>, std::string, std::string, std::function <Eigen::MatrixXd (Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > , std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & )> nn_ptr_);
    forward_prop(uint, uint, uint, uint, std::vector<uint>, std::string, std::string, std::function <Eigen::MatrixXd (Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > , std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & )> nn_ptr_, std::vector<bool>);
    forward_prop(uint, uint, uint, uint, std::vector<uint>, std::string, std::string, std::function <Eigen::MatrixXd (Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > , std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & )> nn_ptr_, std::vector<bool>, std::vector<bool>);
	void setup_LL(std::string);
	double operator()(Eigen::Ref<Eigen::VectorXd> w);
    void test_output(Eigen::Ref<Eigen::VectorXd> w);		
protected:
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
    std::vector<double> y_tr_v;
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > y_tr_m;
    std::string LL_type; 
    double LL_var;
    double LL_norm; 
    const uint num_complete_batches;
    const uint num_batches;
    uint b_c;
    Eigen::Matrix<uint, Eigen::Dynamic, 1> rand_m_ind;
    std::function <double(Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >, Eigen::MatrixXd &, double &, double &, const uint &, const uint &) > LL_ptr; 
    std::function <Eigen::MatrixXd (Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > , std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & )> nn_ptr;
    const uint LL_dim;
    //member methods
	std::vector<uint> get_weight_shapes();
    std::vector<uint> get_weight_shapes(const std::vector<bool> &);
    std::vector<uint> get_weight_shapes(const std::vector<bool> &, const std::vector<bool> &);
	std::vector<double> get_tr_vec(const uint &, const std::string &);
	uint calc_num_data(const uint &);
	std::vector<double> file_2_vec(const uint &, const uint &, const std::string &);
    void get_batches(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x_tr_b, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & y_tr_b);
	std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > get_weight_matrices(Eigen::Ref<Eigen::VectorXd> w);
};