- Apparently running tf in Cython won't give much of a performance boost, as when you run a tf session it runs the C++ code

- Distributed tf example code:
https://stackoverflow.com/questions/37712509/how-to-run-tensorflow-distributed-mnist-example

- Tf can be run across multiple cpus and gpus: https://hub.packtpub.com/distributed-tensorflow-multiple-gpu-server/. Need to find links to code as links in this article are broken

- Tensorflow with gpu only faster than numpy if speedup from calculating chain of operations in gpu is bigger than overhead of sending initial input to gpu. Transfer around cpu much faster than gpu
https://towardsdatascience.com/numpy-vs-tensorflow-speed-on-matrix-calculations-9cbff6b3ce04

- For C++ numerical libraries, for cpu-only I will try Eigen, if I require gpu or multi-cpu/gpu I will try arrayfire or thrust. 
Note other parallel alternatives described here:
https://arrayfire.com/benchmarking-parallel-vector-libraries/
