#include <iostream>
#include <Eigen/Dense> //EIGEN library
#include <unistd.h>

using namespace Eigen;
using namespace std;

int main()
{
	srand(1);
for(int i = 0; i < 5; i++) {

        MatrixXf A = MatrixXf::Random(3, 3);
        cout << A <<endl;
}
}
