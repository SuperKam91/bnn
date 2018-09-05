# bnn
Training Bayesian neural networks using Markov chain Monte Carlo techniques.
# thoughts scratchpad
- look into applications of bnns to deep reinforcement learning- believe they are useful as in rl aren't always looking for optimal value function approximation, having stochastic output helps with exploration of algorithm

- for pure nn applications (i.e. regression and classification) in general if a lot of data is available, can build an ar almost arbitrarily large network these days (thanks to gpus) to perform well, regardless of the complexity of the task. Thus we might want to focus on examples where data is scarce. In these cases, building complex neural networks could lead to overfitting, thus simple networks which perform well (hopefully bnn) should be optimal

- I also suspect that bnns are more useful for regression than classification. In the latter case, MLE/MPE already gives a probability for each class output, thus a bnn is just giving a 'probability distribution of probabilities', which in isolation isn't very satisfying (though the statistics one can calculate from these distributions are still interesting). On the other hand in regression, having a probability distribution over some output variable will almost always be useful

- might be best to use tensorflow to build forward propagation computation graph, have this output the cost, and feed this into polychord (or pass function which runs forward computation to polychord, will likely need a wrapper to calculate likelihood value from cost). Can first try this idea using python tf, then if it works, go over to c++ tf for efficiency.

- will probably need to ignore all explicit regularisation techniques (i.e l^n norm regularisation) as polychord handles prior separately from likelihood/cost function. OR could handle prior in cost function using regularisation, then assign uniform priors in polychord.

- need to look at how tf is parallelised in both python and c++

- need to think how batches will be handled with polychord (opposed to passing all data at once)

- need to think how to pass weights sampled from polychord to tf to evaluate forward prop. Should be simple enough with placeholders and feed methodology
