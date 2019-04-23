#include <iostream>
#include <Eigen/Dense> //EIGEN library

using namespace Eigen;
using namespace std;

int main()
{
MatrixXd A;
A.setRandom(10000, 10000);

MatrixXd B;
B.setRandom(10000, 10000);

MatrixXd C;
C=A*B;
}