# memory release

- when a variable goes out of scope the name of the variable is wiped and the memory associated with the variable is released. The same goes for arrays on the stack. However for dynamic arrays (i.e. on the heap), the memory is not released without using the delete keyword, and so if the name goes out of scope without releasing the memory, this memory in general cannot be recovered. Also, trying to delete a name associated with the same memory more than once can have undefined results.

# compiler explorer: https://gcc.godbolt.org/

- note in general how variables/arrays are stored is compiler dependent.

- for how global (also applies for static variables I think) variables and arrays are allocated and saved in the binary during compilation, see https://stackoverflow.com/questions/21350478/what-does-memory-allocated-at-compile-time-really-mean and specifically, manu's, supercat's, Elias Van Ootegem's and mah's answers. Also, run code compiler_explorer/comp_memory_explorer.cpp in compiler explorer. Note this example shows dynamic arrays are not assigned required memory at compilation time, as they are constantly assigned 8 bytes regardless of their size. Also, array only increases size of binary by its size if it is initialised.

- for local variables, the situation is much more complicated. Again, see compiler_explorer/comp_memory_explorer.cpp in compiler explorer.

- classes also complicated, but their methods seem to act similar to functions. Again see comp_memory_explorer.cpp.

# types and keywords

- Static variables when used inside functions are initialised only once, and then they hold their value even through function calls. Static elements are allocated storage only once in a program lifetime in static storage area. 

# function returns

- functions by default return by value. this means when a variable is returned, a copy of the value returned in the called function is made outside of the function being called. (i.e. copy outside of function call has different address to returned value)

- can return by reference, but local variables on stack are destroyed after called function exits, leading to dangling pointers (can be avoided by allocating to heap).

- some objects (e.g. vectors) are effectively constructed directly in place in the caller if the compiler is allowed to do named return value optimisation. this works as follows: the caller reserves some space for return value, then passes address to calling function. calling function then uses this address for return value. means that the return value and the local variable being returned (if return value is simply the local variable) share the same memory i.e. reference the same thing. 

- eigen maps don't seem to do nrvo

# passing to function by value

- hierarchical structures e.g. vector of vectors, vector of eigen maps may or may not reference the same memory at the "bottom" of the structure. for example, if passing a vector of eigen maps by value, the addresses of the maps will be different between the copies, but the maps will reference the same underlying memory. makes sense if you consider a map just containing the address of what it references (as this value is copied to the next vector). however when a vector of vectors storing ints is copied, the bottom underlying memory storage addresses are different.

# placement new

- https://www.geeksforgeeks.org/placement-new-operator-cpp/