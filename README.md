# bnn
Training Bayesian neural networks using Markov chain Monte Carlo techniques.
# timeline
- Implement support for stochastic hyperparameters (prior and likelihood)
- Compare Bayesian and MLP estimates for small datasets, where using small networks compatible with PolyChord 1 is justifiable. e.g. Boston housing dataset regression, or FIFA World Cup 2018 classification.  
- Look into tradeoff of using tensorflow with gpu support vs c++ implemented nns (possibly incorporating parallelisation) vs cython

# background reading

- Bayesian sparse reconstruction: a brute-force approach to astronomical imaging and machine learning
https://arxiv.org/abs/1809.04598

- Theses which focus on bnns and their applications, to draw inspiration from: http://mlg.eng.cam.ac.uk/yarin/blog_2248.html#thesis (Gal, 2016), www.cs.ubc.ca/~nando/papers/thesis.pdf (Freitas, 1999), http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.446.9306&rep=rep1&type=pdf (Neal, 1995), http://www.inference.org.uk/mackay/PhD.html (Mackay, 1991) 

- bnn workshop @ nips. Contains extended abstracts and videos to some of workshops:
http://bayesiandeeplearning.org/

- nips general pages. Can find more talks (not in workshops) here:
https://nips.cc/Conferences/2018

- some general bnn papers:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.274&rep=rep1&type=pdf (1992),
https://papers.nips.cc/paper/613-bayesian-learning-via-stochastic-dynamics.pdf (1992),
https://www.tandfonline.com/doi/abs/10.1088/0954-898X_6_3_011 (1995),
https://www.nature.com/articles/nature14541.pdf (2015)

- talks on bnns from icml 2018 (9):
https://icml.cc/Conferences/2018/Schedule?showParentSession=3377,
https://icml.cc/Conferences/2018/Schedule?showParentSession=3437, 
https://icml.cc/Conferences/2018/Schedule?showParentSession=3447

- talks on bnns from icml 2017: (3)
http://proceedings.mlr.press/v70/gao17a.html,
http://proceedings.mlr.press/v70/li17a.html,
http://proceedings.mlr.press/v70/louizos17a.html

- talks on bnns from icml 2016 (1):
http://proceedings.mlr.press/v48/gal16.pdf

- talks on bnns from icml 2015 (2):
http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf,
https://arxiv.org/pdf/1505.05424.pdf

- earliest resources relating to bnns:
https://pdfs.semanticscholar.org/33fd/c91c520b54e097f5e09fae1cfc94793fbfcf.pdf (1987), https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=118274 (1989, not worth printing), https://papers.nips.cc/paper/419-transforming-neural-net-output-levels-to-probability-distributions.pdf (1991), https://pdfs.semanticscholar.org/c836/84f6207697c12850db423fd9747572cf1784.pdf (1991)

- talks on bnns:
http://bayesiandeeplearning.org/2016/slides/nips16bayesdeep.pdf (2016)

- variational inference and bnns:
http://www.cs.toronto.edu/~fritz/absps/colt93.pdf (1993), https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ensemble-nato-98.pdf (1998), https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks.pdf (2011)

- misc papers relating to bnns:
https://arxiv.org/pdf/1511.03243.pdf (2015),
https://arxiv.org/pdf/1512.05287.pdf (2016),
https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf (2011)

# thoughts scratchpad

- Transfer learning is a potential opportunity: train all but final layer using MLE methods (or use pre-trained weights) to e.g. learn featuring generating nodes. Train final layer(s) with BNN to learn accurate representation of non-linear combination of these learned features, and some uncertainty. Problem is, previous layers cannot be too big, or even training the final layers will not be possible. Unfortunately this will often be the case with CNNs, as the number of channels usually increases with depth into the NN (as the CNN learns a larger number of intricate features as one delves into the NN).

- An alternative to the above is using MLE/MPE methods to train an (variational) autoencoder to reduce the dimensions of the input, and then feed this to a BNN. Downside of this is you will lose some information in the encoding of the input, which may outweight the gains of treating the problem in a Bayesian framework

- Seems that there are applications in ML in areas such as medicine, where number of training data is small. Thus one is forced to use small networks to prevent overfitting. See bullet below.

- https://reader.elsevier.com/reader/sd/pii/S0933365716301749?token=D3ADCF7EC51CE6EBC44A540012EAAFB1F5D8CFC8BA12B847B380A83E838D1F4444E61168F6FA06D9F43A5D794D04504D describes a regression problem with five input features and 35 records. Use a one layer NN (~40 parameters), but train over 20000 NNs (with different parameter initialisations, and different randomisations of data for stochastic gradient descent), and pick NN with best performance on cross-validation set, for their final evaluation. Use RMSE and R^2 regression coefficient to evaluate performance, and get values RMSE = 0.85 and R^2 = 0.983 for final evaluation. I believe they run the algorithms for ~30 epochs.

- Mackay uses Gaussian approximation to get solutions to BNNs. Picks maximum likelihood vlaue of hyperparameters (using analytical solution I believe), then looks at evidence values and training set error to evaluate models.

- Neal uses HMC to sample the BNN parameters, and Gibbs sampling to sample the hyperparameters. n.b. HMC requires gradient information, so can't be used to sample hyperparameters directly (to my knowledge). Also, HMC in essence has characteristics similar to common optimisation methods which use 'momentum' and 'velocity'.

- Neal also introduces concept of using Gaussian processes to introduce a prior over functions, which tells us what nn predicts mapping function to be without any data.

- sampling hyperparameters seems rather necessary to justify allowing NN to be arbitrarily big- if complex model is not needed, hyperparameters will 'quash' nodes which aren't important, according to the hyperparameter values assigned by the data during the training, and 'upweight' important nodes. Also avoids cross validation step.

- seems BNNs are useful in approximating Q(s,a) function in RL- rather than approximating maximum of Q(s,a) for values of states and actions, better to have a probability distribution over each Q(s,a), so a can be sampled. Sample from each Q(s,a) for a given state, and choose a corresponding to highest Q value. The stochasticity introduced in sampling a supposedly helps the exploration of the algorithm. see: https://www.coursera.org/learn/practical-rl/lecture/okvvc/thompson-sampling

- look into applications of bnns to deep reinforcement learning- believe they are useful as in rl aren't always looking for optimal value function approximation, having stochastic output helps with exploration of algorithm.

- for pure nn applications (i.e. regression and classification) in general if a lot of data is available, can build an almost arbitrarily large network these days (thanks to gpus) to perform well, regardless of the complexity of the task. Thus we might want to focus on examples where data is scarce. In these cases, building complex neural networks could lead to overfitting, thus simple networks which perform well (hopefully bnn) should be optimal

- I also suspect that bnns are more useful for regression than classification. In the latter case, MLE/MPE already gives a probability for each class output, thus a bnn is just giving a 'probability distribution of probabilities', which in isolation isn't very satisfying (though the statistics one can calculate from these distributions are still interesting). On the other hand in regression, having a probability distribution over some output variable will almost always be useful

- will probably need to ignore all explicit regularisation techniques (i.e l^n norm regularisation) as polychord handles prior separately from likelihood/cost function. OR could handle prior in cost function using regularisation, then assign uniform priors in polychord

- distributed tf uses master & slave and mapreduce paradigms. need to see if we can use distributed tf on cambridge hpc

- tf and keras can use gpus almost automatically. very little intervention needed. need to see if we can use tf-gpu on cambridge hpc gpu systems

- need to think how batches will be handled with polychord (opposed to passing all data at once)

- one idea is to use tf/keras optimisation a little bit to get a good 'initial guess' of the network parameters. pass these to polychord and it can use these as the initial livepoints. note however these initial livepoints would have to be independent. subsequent samples will be at likelihoods higher than the initial set, and so should be able to concentrate on sampling the peak of the posterior. avoids wasting time sampling 'unreasonable' network parameters. another problem with this is the initial livepoints aren't sampled according to the prior, unless we implement the prior (regularise) in the optimisation exploration (then can include prior in ll func to pc with uniform priors or treat priors separately as normal)
