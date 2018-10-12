- #include <Eigen/Core> should be sufficient for all matrix and array classes, as well as basic linear algebra operations.

- matrix/ vectors are either fixed size (e.g. m = Matrix<typename Scalar, int i, int j>) or dynamic (e.g. m = MatrixXf(i,j)), the former is usually stored on the stack while the former uses dynamic memory allocation (usually heap). Fixed size really amounts to just declaring float mymatrix[i*j]; while dynamic size is more like float *mymatrix = new float[i*j]. note when a matrix size is defined during its definition (e.g. Matrix4f m(5,5)) it is still a dynamic matrix.

- eigen objects have maximum dimension of 2, for higher dimensions one must stack these into two dimensions, or form array of eigen objects.

- fixed size matrices must have their size determined at compile time, but gain an efficiency boost for small matrices (<16). Using fixed size for large vectors can lead to stack overflow, and possibly a loss of performance over dynamic sizes due to how eigen vectorises an operation (n.b. in general stack is faster to access than heap, but latter can have total size adjusted whereas stack is fixed, so stackoverflow can occur for large matrices).

- dynamic matrices can be resized, but generally this acts as a destructor on the object and thus leads to loss of the values in the original, unless a special 'conservative' resize is performed.

- eigen doesn't do automatic type promotion: when operating on two or more eigen objects their types should be the same.

- arithmetic operators such as + - / * aren't performed until the whole expression (line) is evaluated, typically using the = operator. This is so that each object only needs to be iterated through once per line for a performance boost. this encourages one to do heavy arithmetic expressions in one line. e.g.
	a = 3*b + 4*c + 5*d is performed as for (int i=0,i<len,++i){a[i] = 3*b[i] + 4*c[i] + 5*d[i];}

- eigen checks validity of operations at compile time if possible. however when using dynamic matrices, checks often cannot be performed until runtime (dynamic memory allocation isn't performed until runtime, think this means the size of the object isn't known until then), in which case eigen uses runtime assertions. n.b. program will probably crash if assertions are turned off. consequently even if you state the object size during declaration, it will only be checked during runtime. 

- arrays are another eigen object (declared by e.g. ArrayXXf a(3,3)) which support element-wise operations and addition/subtraction of scalars.

- matrix objects have a .array() method which 'converts' them to arrays and vice versa for array objects, but note expressions involving both types is forbidden, the one exception being the assignment (=) operator. e.g. mprod = m1.array() * m2.array() calculates the element-wise product of m1 and m2 (by converting them to arrays) and stores the result in matrix mprod. provided you let your compiler optimise, converting between them has no overhead cost.

- the block method for matrices or arrays (for block of size (p,q), starting at (i,j) matrix.block(i,j,p,q) for dynamic block and matrix.block<p,q>(i,j) for fixed block) returns a block from the matrix. note to use fixed size block, its size must be known at compile time. matrix.row(i) retrieves ith row of matrix, similar for .col()

- for vectors there are equivalent methods e.g. v.segment(i,k) is the operation which returns the dynamic block representing the first k elements of v starting from index i.

- even if object is dynamic, should give eigen as much information as possible at compile time so it can try to optimize.

- map objects can be used to work with "raw" C++ arrays such that the memory occupied by the raw array is directly used by the eigen object, so that it doesn't need to be copied.

- to construct a map variable you need two pieces of information about the data: a pointer to the region of memory containing that data, and the desired shape of the matrix or vector e.g. Map<MatrixXF> mf(pf, rows, columns) where pf is a pointer to the data. Note map type objects are not the same as their dense counterparts, this is particularly important when writing your own functions which work with eigen objects.

- matrix.unaryExpr(std::ptr_fun(func)) applies func to matrix element-wise. However for simpler functions supported by eigen e.g. exp(): matrix.array().exp() works

# troubleshooting tips:

- https://github.com/ethz-asl/eigen_catkin/wiki/Eigen-Memory-Issues
