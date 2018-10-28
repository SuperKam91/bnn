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
scratch_dir = $(source_dir)/scratch
external_dir = $(PWD)/external
include_dir = $(PWD)/include
build_dir = $(PWD)/build
binary_dir = $(PWD)/bin
lib_dir = $(PWD)/lib


# Find files
srcs = $(wildcard $(source_dir)/*.cpp)
objs = $(patsubst %, $(build_dir)/%, $(notdir $(srcs:%.cpp=%.o)))
deps = $(patsubst %, $(build_dir)/%, $(notdir $(srcs:%.cpp=%.d)))
test_srcs = $(wildcard $(test_dir)/*.cpp)
test_objs = $(patsubst %, $(build_dir)/%, $(notdir $(test_srcs:%.cpp=%.o)))
test_deps = $(patsubst %, $(build_dir)/%, $(notdir $(test_srcs:%.cpp=%.d)))
scratch_srcs = $(wildcard $(scratch_dir)/*.cpp)
scratch_objs = $(patsubst %, $(build_dir)/%, $(notdir $(scratch_srcs:%.cpp=%.o)))
scratch_deps = $(patsubst %, $(build_dir)/%, $(notdir $(scratch_srcs:%.cpp=%.d)))
# Extra flags
inc += -isystem$(external_dir) -I$(include_dir)
LDSHARED = $(LD) -shared
LDLIBS +=  -l$(libname) #not currently used
LDFLAGS += -L$(lib_dir) -lchord 

all: main
extra: tags doc
main: $(binary_dir)/main 
test: $(binary_dir)/test 
scratch: $(binary_dir)/scratch

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

# Compiling the test_main program
$(binary_dir)/test: $(build_dir)/test_main.o $(objs) $(test_objs)
	@mkdir -p $(@D)
	$(LD) $^ -o $@ $(LDFLAGS)

# Compiling the scratch program
$(binary_dir)/scratch: $(build_dir)/scratch_main.o $(scratch_objs)
	@mkdir -p $(@D)
	$(LD) $^ -o $@ $(LDFLAGS)

# Building main object file and dependency
$(build_dir)/main.o: main.cpp $(makefiles)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@

# Build object and dependency files from src/
$(build_dir)/%.o: $(source_dir)/%.cpp 
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@

# Building test_main object file and dependency
$(build_dir)/test_main.o: test_main.cpp 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@

# Build object and dependency files from src/test/
$(build_dir)/%.o: $(test_dir)/%.cpp 
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@

# Building scratch_main object file and dependency
$(build_dir)/scratch_main.o: scratch_main.cpp 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@

# Build object and dependency files from src/scratch/
$(build_dir)/%.o: $(scratch_dir)/%.cpp 
	$(CXX) $(CXXFLAGS) $(inc) -MMD -c $< -o $@
	
# Build c src/ files
$(build_dir)/%.o: $(source_dir)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(inc) -MMD -c $< -o $@

all_srcs = $(shell find src -name '*.[ch]pp') #possibly should be src

# Build tags file for vim
tags: $(all_srcs)
	ctags --extra=+f $(all_srcs)

.PHONY: clean main test

clean:
	$(RM) -r $(objs) $(deps)

purge: clean
	$(RM) -r $(build_dir) $(lib_dir)
	$(RM) -r __pycache__ doc/html
	$(RM) tags 

# Include the dependencies
-include $(deps)
