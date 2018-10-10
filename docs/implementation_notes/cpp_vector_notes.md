- like arrays, vectors store data in contiguous memory blocks. unlike arrays, the size of a vector can change dynamically.

- internally, vectors use dynamically allocated arrays to store data. The storage allocated to the array is usually higher than the amount currently required by the vector's elements. This allows one to add more elements to the vector without having to re-allocate it to a larger block of memory. If a relatively large increase in vector size is required, the 'wiggle room' associated with the extra space in the dynamic array will not be sufficient, and a re-allocation is necessary. re-allocations are an inefficient process as data must be copied from the old memory location to the new one, and so there is a trade off between having too much unused memory in the dynamic array (i.e. size(dynamic array) >> size(required by vector)) and having the dynamic array not have much wiggle room, but a high chance of re-allocation if the vector needs to store more data.

- vectors are very efficient at accessing their elements (like arrays), and quite efficient at adding or removing elements from their end points. However they are quite inefficient at operations which involve removing or adding elements not at their end points.

- although based on dynamic arrays, vectors do not need to be deleted manually (they are deleted when they go out of scope, thus they shouldn't be manually deleted). to free up the memory of a vector, use the swapping trick described below.

- the method .push_back(x) appends an element to the vector and fills it with x, and can trigger re-allocation if necessary.

- .reserve(n) will grow the allocated storage of the vector to size n if necessary, but will never shrink it (even if previous storage > n). Note it does not initialise any new elements created, but std::vector<type> v(n); would initialise (call the default constructor of the type object) all n elements.

- To find out how many elements would fit in the currently allocated storage of a vector, use the .capacity() member function. To find out how many elements are currently contained by the vector, use the .size() member function. the number of elements that can be added to a vector without triggering a reallocation always is .capacity() - .size()

- to change the number of contained elements in a vector call .resize(n, x). If n is larger than the old size of the vector, it will preserve all elements already present in the controlled sequence; the rest will be initialized to x (or 0 if not specified). If the new size is smaller than the old size, it will preserve only the first new_size elements. The rest is discarded and shouldn't be used any more—consider these elements invalid. If the new size is larger than .capacity(), it will reallocate storage so all new_size elements fit. .resize() will never shrink .capacity(). Note .resize() calls the default constructor for the type of the vector like std::vector<type> v(n) does.

- using .reserve(n) and then .pushback() can be more efficient than using .resize(n) and then using [] or .at()

- reallocation often leads to the vector being stored somewhere else in memory. this means if a pointer is set to the address of a vector, and the vector is reallocated, the pointer will remain pointed to the old location, where the data has been deleted from.

- An iterator is a handle to a contained element. you can dereference it to obtain the element it "points" to. constant iterators cannot be used to modify the contents of the vector. n.b. .end() points to the last+1 element, and so dereferencing has undefined behaviour. reallocating a vector invalidates iterators like it does pointers.

- the assignment operator can be used to copy the elements from one vector to another e.g. v2 = v1 copies the elements of v1 to v2. similarly the copy constructor can be called std::vector<type> v2(v1). The assign() function will reinitialise the vector. We can pass either a valid element range using the [first, last+1] iterators e.g. .assign(v.begin(), v.end()) or we can specify the number of elements to be created and the element value e.g. v.assign(69, 5).

- pop_back() deletes last element from vector, but does not shrink the capacity().

-  rbegin() and rend() point to the first, and last+1 element of the reverse sequence. Note that both rbegin() and rend() return the type reverse_iterator, not iterator. to get the iterator object use reverse_iterator.base().

- .front() and .back() return the first and last element values respectively.

- .insert(it, x) fills the element before it with value x (adds an element in front of it). .insert(it, n, x) inserts x n times before position of it. insert() returns iterator pointing to new element in case of one being added, or the first of the new elements if several are added. .erase(it_first, it_last) removes elements in [it_first, it_last), .erase(deletes all elements). Note none of the memory is freed, and capacity isn't decreased. inserting and erasing elements may invalidate iterators already pointing to the vector, depending on where they point to relative to where the actions were performed.

- performing comparisons (==, >, etc.) of vectors is done on an element-wise basis.

- Sometimes, it is practical to be able to .swap() the contents of two vectors. A common application is forcing a vector to release the memory it holds. crucially, .swap() swaps the .capacity() of the two vectors in question. Thus the following will remove all elements from v, and shrink its .capacity() to zero (or something small): v.clear(); v.swap(std::vector<type>()); v now has the capacity (and elements) of std::vector<type>(), and vice versa. The latter gets destroyed after this statement is executed, and thus the memory originally associated with v is freed, and one is left with a vector v with 0 elements and the same .capacity as std::vector<type>().
