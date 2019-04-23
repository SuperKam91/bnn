#include <iostream>
#include <Eigen/Dense> //EIGEN library

using namespace Eigen;
using namespace std;

int main()
{
MatrixXd A;
A.setRandom(100, 100);

MatrixXd B;
B.setRandom(100, 100);

MatrixXd C;
C=A*B;
}