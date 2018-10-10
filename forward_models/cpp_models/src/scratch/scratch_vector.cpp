/* include external code */
#include <iostream>
#include <vector>

/* include in-house code */
#include "scratch/scratch_vector.hpp"

void scratch_vector()
	{
    std::vector<int> array;
    std::vector<int>::const_iterator i; //iterator for vec
    int arr[] = {6, 7, 8, 9, 10};
    std::vector<int> array2(arr, arr+5); //neat way of initialising vectors. n.b. this copies arr (doens't overwrite)
    int x = 999; 
    unsigned long int size = 5;         // if don't use unsigned long int, throws warning
    array.reserve(size);    // make room for 5 elements
    array.push_back(x);
    std::cout<<array.capacity()<<std::endl; // 5
    std::cout<<array.size()<<std::endl; // 1
    for(unsigned long int j=0; j<size; ++j)
    	{
    	array[j] = int (j); // cast unsigned long int to int as this is type for which vector template was declared
    	//array.push_back(int (j)); // increases size by one
    	//array.at(i) = int (j); //throws std::out_range exception
		}
	std::cout<<array.size()<<std::endl; // still 1 (!) would be 10 if push_back used
	for(i=array.begin(); i!=array.end(); ++i) //iterate over vector using iterator. .begin() points to 1st element, .end() points to last+1	
	{
        std::cout<<(*i)<<std::endl;
    }

	std::vector<int> array3;   // create an empty vector
    array3.reserve(3);         // make room for 3 elements
                              // at this point, capacity() is 3
                              // and size() is 0
    array3.push_back(999);     // append an element
    array3.resize(5);          // resize the vector
                              // at this point, the vector contains
                              // 999, 0, 0, 0, 0
    array3.push_back(333);     // append another element into the vector
                              // at this point, the vector contains
                              // 999, 0, 0, 0, 0, 333
    array3.reserve(1);         // will do nothing, as capacity() > 1
    array3.resize(3);          // at this point, the vector contains
                              // 999, 0, 0
                              // capacity() remains 6
                              // size() is 3
    array3.resize(6, 1);       // resize again, fill up with ones
                              // at this point the vector contains
                              // 999, 0, 0, 1, 1, 1
	}    