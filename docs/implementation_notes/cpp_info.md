# posts explaining what should be included in headers and what should be in source:
https://stackoverflow.com/questions/1945846/what-should-go-into-an-h-file, 
https://stackoverflow.com/questions/9579930/separating-class-code-into-a-header-and-cpp-file, 
https://stackoverflow.com/questions/1809679/difference-between-implementing-a-class-inside-a-h-file-or-in-a-cpp-file

# posts explaining how include files work:
https://stackoverflow.com/questions/333889/why-have-header-files-and-cpp-files, 
https://stackoverflow.com/questions/3098281/can-someone-help-clarify-how-header-files-work

# how to structure classes:
https://stackoverflow.com/questions/9075931/function-declaration-inside-or-outside-the-class

# misc:
http://www.acodersjourney.com/2016/05/top-10-c-header-file-mistakes-and-how-to-fix-them/ (how to stop circular dependencies and others)

# templates:

- Function templates are special functions that can operate with generic types. This allows us to create a function template whose functionality can be adapted to more than one type or class without repeating the entire code for each type.

- To declare a function template use the syntax: template <class type> return_type function_name(...); type can then be used as the type for arguments to the function, the variables within the function, and the return type. The function definition is then template <class type> function_name(...){...} Note a template can have more than one templated type: <class type_1, class type_2> return_type function_name(...)

# function pointers:

- calling a function without parenthesis e.g. int func(); func; returns the address where the function is stored (c++ implicitly converts func into a function pointer)

- function pointers are declared using type (* f_ptr)(types); type is return type of function f_ptr is pointing to and types are types of arguments of function which f_ptr points to. 

- since functions are implicitly function pointers, to evaluate a function using a pointer to it, one simply does f_ptr(args);

-  Default parameters won’t work for functions called through function pointers. Default parameters are resolved at compile-time (that is, if you don’t supply an argument for a defaulted parameter, the compiler substitutes one in for you when the code is compiled). However, function pointers are resolved at run-time.

- function pointers can be used to pass functions (by means of a pointer) to another function e.g. type_2 func_2(type_1(*f_ptr)(types)){...}; func_2(func); where func returns type_1 and has types for its args.

# functors:

- A functor is pretty much just a class which defines the operator(). That lets you create objects which "look like" a function (see scratch_functors.cpp). They can be used just like function pointers

- unlike regular functions, they can contain state. The example add_x creates a function which adds 42 to whatever you give it. But that value 42 is not hardcoded, it was specified as a constructor argument when we created our functor instance. We could create another adder, which added 27, just by calling the constructor with a different value. This makes them nicely customizable.

- One often passes functors as arguments to other functions such as std::transform or the other standard library algorithms. You could do the same with a regular function pointer except, as I said above, functors can be "customized" because they contain state, making them more flexible (If we wanted to use a function pointer, I'd have to write a function which added a specific number to its argument. The functor is general, and adds whatever you initialised it with)

- They are also potentially more efficient when passed to other functions, as the compiler knows exactly which function the function being passed to should call. It should call add_x::operator(). That means it can inline that function call. And that makes it just as efficient as if we had manually called the function ourselves. If we had passed a function pointer instead, the compiler couldn't immediately see which function it points to, so unless it performs some fairly complex global optimizations, it'd have to dereference the pointer at runtime, and then make the call.

# smart pointers:

- pointers to object stored on heap (but the pointer itself is stored on the stack I think) which don't need to be deleted manually, to prevent memory leakage. from c++11 onwards there are three (useful and relatively non-lethal) types of smart pointers included in std: std::unique_ptr, std::shared_ptr and std::weak_ptr. 

- unique_ptr is the simplest case, and is necessary when one requires a single pointer to an object. unique_ptrs cannot be copied, which prevents the object being deleted (released from memory) several times. You can, however, pass references to it around to other functions you call.

- A more complex smart pointer policy involves reference counting the pointer. This does allow the pointer to be copied (ptr2 = ptr1). When the last "reference" to the object is destroyed, the object is deleted. this is implemented through std::shared_ptr. useful when you want pointers defined in different scopes to reference same object.

- if smart pointers (this may only apply to std::shared_ptr, need to check) are declared on heap (that is, std::shared_ptr<MyObject> = new std::shared_ptr<MyObject>(new <MyObject>)), I believe they need to be deleted manually to delete MyObject.

- another possible lead to memory leakage is circular references in the case of std::shared_ptrs (i.e. when the shared_ptrs reference each other). std::weak_ptr is a (uncounted) way to reference a shared_ptr so that its object is deleted without needing the count to go to zero.

# initialisation using () [direct initialisation] versus using = [copy initialisation]:

- class b(a); is not strictly the same as class b = a; depending on the compiler (whether it uses elision optimisation), and the type of a. See: https://stackoverflow.com/questions/587070/c-constructor-syntax/587116#comment14499992_587116, https://stackoverflow.com/questions/637259/initializing-which-one-is-more-efficient, https://stackoverflow.com/questions/1051379/is-there-a-difference-between-copy-initialization-and-direct-initialization, https://stackoverflow.com/questions/4470553/initialization-parenthesis-vs-equals-sign

