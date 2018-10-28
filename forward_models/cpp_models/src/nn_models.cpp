/* external codebase */
#include <vector>
#include <Eigen/Dense>

/* in-house code */
#include "nn_models.hpp"
#include "mathematics.hpp"

Eigen::MatrixXd slp_nn_1(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = (x * w[0]).array().tanh();
    Eigen::MatrixXd a2 = a1.rowwise().sum();
    return a2;
}

//struggling to think of way to avoid creating temporary std::ptr_fun upon every call without making it global,
//or converting nn functions into functors and saving as member variable (but then forward prop would have to have different kind of 
//pointer pointing to nn functor). could use same style as inverse priors: base_prior with pure virt operator(), then have every nn
//derive from it, and in forward_prop const create base ptr to dynamically allocated derived nn obj and in forward_prop operator() evaluate base_ptr operator()
Eigen::MatrixXd slp_nn_old(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose()).unaryExpr(std::ptr_fun(relu));
    Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[3].data(), w[3].size()).transpose());
    return a2;
}