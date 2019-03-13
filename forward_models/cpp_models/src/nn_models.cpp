/* external codebase */
#include <vector>
#include <Eigen/Dense>
#include <iostream>

/* in-house code */
#include "nn_models.hpp"
#include "mathematics.hpp"

Eigen::MatrixXd slp_sm(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    //I think row-wise implementations of activations in other nns is just for efficiency. here I will be more general and use col-wise
    Eigen::MatrixXd z1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose());
    return softmax(z1);
}

Eigen::MatrixXd slp(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    //I think row-wise implementations of activations in other nns is just for efficiency. here I will be more general and use col-wise
    Eigen::MatrixXd z1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose());
    return z1;
}

Eigen::MatrixXd mlp_test(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = (x * w[0]).array().tanh();
    Eigen::MatrixXd a2 = a1.rowwise().sum();
    return a2;
}

//struggling to think of way to avoid creating temporary std::ptr_fun upon every call without making it global,
//or converting nn functions into functors and saving as member variable (but then forward prop would have to have different kind of 
//pointer pointing to nn functor). could use same style as inverse priors: base_prior with pure virt operator(), then have every nn
//derive from it, and in forward_prop const create base ptr to dynamically allocated derived nn obj and in forward_prop operator() evaluate base_ptr operator()
Eigen::MatrixXd mlp_test2(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((x * w[0]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[1].data(), w[1].size())).transpose()).unaryExpr(std::ptr_fun(relu));
    Eigen::MatrixXd a2 = (a1 * w[2]).rowwise() + (Eigen::Map< Eigen::VectorXd> (w[3].data(), w[3].size()).transpose());
    return a2;
}

Eigen::MatrixXd mlp_uap_ResNet_1(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    uint num_blocks = 1;
    uint weight_slice = 3;
    Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a0 = x;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a2;
    for (uint i = 0; i < num_blocks; ++i) {
    std::cout<<w.size()<<std::endl;
        a2 = mlp_uap_ResNet_block(a0, w.at(i * weight_slice), w.at((i * weight_slice) + 1), w.at((i * weight_slice) + 2));
        a0 = a2;
    }
    Eigen::MatrixXd output = a2; //convert to columnwise for returning value. i.e. to be consistent with other nn_models. creates a copy of a2 to do this, however.
    return output;
}

Eigen::MatrixXd mlp_coursera_ResNet_1(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    uint num_blocks = 1;
    uint weight_slice = 4;
    Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a0 = x;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a2;
    for (uint i = 0; i < num_blocks; ++i) {
        a2 = mlp_coursera_ResNet_block(a0, w.at(i * weight_slice), w.at((i * weight_slice) + 1), w.at((i * weight_slice) + 2), w.at((i * weight_slice) + 3));
        a0 = a2;
    }
    Eigen::MatrixXd output = a2;
    return output;
}

//as with python implementation, could easily make functions for general num_blocks, but cba init as wouldn't be able to specify from main without external var or without reconfiguring forward class
Eigen::MatrixXd mlp_uap_ResNet_2(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    uint num_blocks = 2;
    uint weight_slice = 3;
    //may be more efficient way of doing this using a ref or map if can find out how to change what they point to 
    //(i.e. change them from pointing to x to point to a2 in loop). apparently can do it with map and the new assignment operator,
    //but have had bad experiences of using this in the past. 
    //with current implementation, have to copy x over to a0
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a0 = x;
    //map implementation (more efficient as doesn't have to copy x) ACTUALLY SEEMS TO WORK, SO FAR, SO WILL GO WITH IT.
    //------------------------------------------------------------------------------------------------------------------
    long int row_num = x.rows();
    long int col_num = x.cols();
    Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a0(x.data(), row_num, col_num);
    //------------------------------------------------------------------------------------------------------------------
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a2;
    for (uint i = 0; i < num_blocks; ++i) {
        a2 = mlp_uap_ResNet_block(a0, w.at(i * weight_slice), w.at((i * weight_slice) + 1), w.at((i * weight_slice) + 2));
        // a0 = a2;
        new (&a0) Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > (a2.data(), row_num, col_num);
    }
    Eigen::MatrixXd output = a2; //convert to columnwise for returning value to be consistent with other nn_models. creates a copy of a2 to do this, however. 
    //may be more efficient to do whole thing columnwise, but not sure this would work since x is rowwise, and thus a0 map is originally rowwise, but would have to be changed to columnwise when changed to point to a2. 
    //probably not worth investigating
    return output;
}

Eigen::MatrixXd mlp_coursera_ResNet_2(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > x, std::vector<Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > > & w) {
    uint num_blocks = 2;
    uint weight_slice = 4;
    Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a0 = x;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a2;
    for (uint i = 0; i < num_blocks; ++i) {
        a2 = mlp_coursera_ResNet_block(a0, w.at(i * weight_slice), w.at((i * weight_slice) + 1), w.at((i * weight_slice) + 2), w.at((i * weight_slice) + 3));
        a0 = a2;
    }
    Eigen::MatrixXd output = a2;
    return output;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mlp_uap_ResNet_block(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a0, Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w1, Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w2, Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w3) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((a0 * w1).rowwise() + (Eigen::Map< Eigen::VectorXd> (w2.data(), w2.size())).transpose()).unaryExpr(std::ptr_fun(relu));
    return a1 * w3 + a0;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mlp_coursera_ResNet_block(Eigen::Ref <Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a0, Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w1, Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w2, Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w3, Eigen::Ref < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > w4) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a1 = ((a0 * w1).rowwise() + (Eigen::Map< Eigen::VectorXd> (w2.data(), w2.size())).transpose()).unaryExpr(std::ptr_fun(relu));
    return (((a1 * w3).rowwise() + (Eigen::Map< Eigen::VectorXd> (w4.data(), w4.size())).transpose()) + a0).unaryExpr(std::ptr_fun(relu));
}