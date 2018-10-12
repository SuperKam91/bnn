/* external codebase */
#include <vector>
#include <Eigen/Dense>

/* in-house code */
#include "nn_models.hpp"
#include "mathematics.hpp"

Eigen::MatrixXd slp_nn_1(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose()).unaryExpr(std::ptr_fun(relu));
    Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[3].data(), w[3].size()).transpose());
    return a2;
}