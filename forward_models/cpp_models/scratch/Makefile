#Kam's MakeFile slightly adapted from Will's on 19/09/18
# Name of libraries
libname = my_project

# Choose OS -specific flags
UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
include Makefile_gnu
else ifeq ($(UNAME), Darwin) 
include Makefile_osx
endif
makefiles = $(wildcard Makefile*)

# Directory definitions
source_dir = $(PWD)/src
test_dir = $(source_dir)/test
external_dir = $(PWD)/external
include_dir = $(PWD)/include
build_dir = $(PWD)/build
binary_dir = $(PWD)/bin
lib_dir = $(PWD)/lib


# Find files
src = $(wildcard $(source_dir)/*.cpp)
objs = $(build_dir)/$(notdir $(src:%.cpp=%.o))
deps = $(build_dir)/$(notdir $(src:%.cpp=%.d))
test_srcs = $(wildcard $(test_dir)/*.cpp)
test_objs = $(test_srcs:%.cpp=$(build_dir)/%.o)
test_deps = $(test_srcs:%.cpp=$(build_dir)/%.d)

# Extra flags
inc += -isystem$(external_dir) -I$(PWD)/include
LDSHARED = $(LD) -shared
LDLIBS +=  -l$(libname)

all: main
extra: tags doc
main: $(binary_dir)/main 
test: $(binary_dir)/test 

$(lib_dir)/lib$(libname).so: $(objs)
	@mkdir -p $(@D)
	$(LDSHARED) $^ -o $@ $(LDFLAGS)

$(lib_dir)/lib$(libname).a: $(objs)
	@mkdir -p $(@D)
	$(AR) $(ARFLAGS) $@ $^ 

python $(libdir)/$(libname).so:
	python3 setup.py install --user
	python2 setup.py install --user

# Compiling the main program
$(binary_dir)/main: $(build_dir)/main.o $(objs)
	@mkdir -p $(@D)
	$(LD) $^ -o $@ $(LDFLAGS)

$(binary_dir)/test: $(build_dir)/test_main.o $(objs) $(test_objs)
	@mkdir -p $(@D)
	$(LD) $^ -o $@ $(LDFLAGS)

# Building main object file and dependency
$(build_dir)/main.o: main.cpp $(makefiles)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@

# Build object and dependency files from src/
$(build_dir)/%.o: $(source_dir)/%.cpp $(makefiles)
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@
	
# Build c src/ files
$(build_dir)/%.o: $(source_dir)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(inc) -MMD -c $< -o $@


all_srcs = $(shell find src -name '*.[ch]pp')

# Build tags file for vim
tags: $(all_srcs)
	ctags --extra=+f $(all_srcs)

.PHONY: clean main test

clean:
	$(RM) $(objs) $(deps) main

purge: clean
	$(RM) -r $(build_dir) $(lib_dir)
	$(RM) -r __pycache__ doc/html
	$(RM) tags 

# Include the dependencies
-include $(deps)
