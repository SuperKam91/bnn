- for a slp nn with 5 hidden units, a Gaussian likelihood, and various input, output and m sizes, tf likelihood class seems to be roughly 3 times faster than Keras. Numpy is 5-10 times faster than tf. This is for cpu only tf, run on local rubbish laptop. can run these with python interfaced version of polychord.

- Apparently running tf in Cython won't give much of a performance boost, as when you run a tf session it runs the C++ code

- Distributed tf example code:
https://stackoverflow.com/questions/37712509/how-to-run-tensorflow-distributed-mnist-example

- Tf can be run across multiple cpus and gpus: https://hub.packtpub.com/distributed-tensorflow-multiple-gpu-server/. Need to find links to code as links in this article are broken

- Tensorflow with gpu only faster than numpy (which only uses cpu) if speedup from calculating chain of operations in gpu is bigger than overhead of sending initial input to gpu. Transfer around cpu much faster than gpu
https://towardsdatascience.com/numpy-vs-tensorflow-speed-on-matrix-calculations-9cbff6b3ce04

- can build own nn model in c++ to use with native polychord c++ interface, or with either a tf graph built in python and exported to C++, or a graph created in C++. there are reports that creating a graph in python the exporting to C++ makes running the graph slow. there are some reports that say this only occurs for the first few runs of the graph, while others say it is consistent. there are also claims that it depends on the network architecture. I guess the only way to know for sure is to compare run times of the python polychord interface using python graphs, exporting graphs to C++, and creating the graphs in C++.
Link to exporting graphs and running in C++:
https://medium.com/@hamedmp/exporting-trained-tensorflow-models-to-c-the-right-way-cf24b609d183
Link to creating graph and running in C++:
https://matrices.io/training-a-deep-neural-network-using-only-tensorflow-c/

- For C++ numerical libraries, for cpu-only I will try Eigen, if I require gpu or multi-cpu/gpu I will try arrayfire, viennacl, blaze-lib or thrust. 
Note other parallel alternatives described here:
https://arrayfire.com/benchmarking-parallel-vector-libraries/
