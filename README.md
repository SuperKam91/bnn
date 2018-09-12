# bnn
Training Bayesian neural networks using Markov chain Monte Carlo techniques.
# thoughts scratchpad
- look into applications of bnns to deep reinforcement learning- believe they are useful as in rl aren't always looking for optimal value function approximation, having stochastic output helps with exploration of algorithm

- for pure nn applications (i.e. regression and classification) in general if a lot of data is available, can build an almost arbitrarily large network these days (thanks to gpus) to perform well, regardless of the complexity of the task. Thus we might want to focus on examples where data is scarce. In these cases, building complex neural networks could lead to overfitting, thus simple networks which perform well (hopefully bnn) should be optimal

- I also suspect that bnns are more useful for regression than classification. In the latter case, MLE/MPE already gives a probability for each class output, thus a bnn is just giving a 'probability distribution of probabilities', which in isolation isn't very satisfying (though the statistics one can calculate from these distributions are still interesting). On the other hand in regression, having a probability distribution over some output variable will almost always be useful

- will probably need to ignore all explicit regularisation techniques (i.e l^n norm regularisation) as polychord handles prior separately from likelihood/cost function. OR could handle prior in cost function using regularisation, then assign uniform priors in polychord

- distributed tf uses master & slave and mapreduce paradigms. need to see if we can use distributed tf on cambridge hpc

- tf and keras can use gpus almost automatically. very little intervention needed. need to see if we can use tf-gpu on cambridge hpc gpu systems

- need to think how batches will be handled with polychord (opposed to passing all data at once)

- one idea is to use tf/keras optimisation a little bit to get a good 'initial guess' of the network parameters. pass these to polychord and it can use these as the initial livepoints. note however these initial livepoints would have to be independent. subsequent samples will be at likelihoods higher than the initial set, and so should be able to concentrate on sampling the peak of the posterior. avoids wasting time sampling 'unreasonable' network parameters. another problem with this is the initial livepoints aren't sampled according to the prior, unless we implement the prior (regularise) in the optimisation exploration (then can include prior in ll func to pc with uniform priors or treat priors separately as normal)

- with regard to the python-C++ interface, first method will be to use python polychord wrapper with keras/tf class passed to it as likelihood function

- second method will be to use c++ polychord interface, with either a tf graph built in python and exported to C++, or a graph created in C++. there are reports that creating a graph in python the exporting to C++ makes running the graph slow. there are some reports that say this only occurs for the first few runs of the graph, while others say it is consistent. there are also claims that it depends on the network architecture. I guess the only way to know for sure is to compare run times of the python polychord interface using python graphs, exporting graphs to C++, and creating the graphs in C++.
Link to exporting graphs and running in C++:
https://medium.com/@hamedmp/exporting-trained-tensorflow-models-to-c-the-right-way-cf24b609d183
Link to creating graph and running in C++:
https://matrices.io/training-a-deep-neural-network-using-only-tensorflow-c/
