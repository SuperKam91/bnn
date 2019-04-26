#include <iostream>
#include <Eigen/Dense> //EIGEN library
#include <unistd.h>

using namespace Eigen;
using namespace std;

int main()
{
unsigned int microseconds = 5e6;

MatrixXd A;
A.setRandom(1000, 1000);

MatrixXd B;
B.setRandom(1000, 1000);

usleep(microseconds);


MatrixXd C;
C=A*B;
usleep(microseconds);
}