# bnn
Training Bayesian neural networks using Markov chain Monte Carlo techniques.

Qualitative document giving an overview of BNNs and what is included in our implementation:
`docs/BNN_overview.pdf`

Currently runs with PolyChordLite version 1.15. e.g. checkout 316effd815b2da5bafa66cfd0388a7c601eaa21d

# Repo guide

The data used in our BNN experiments (in particular the training/test splits) can be found in the `data` repo.

The code which implements our BNNs can be found in the `forward_models` directory. The `Python` implementations of the BNNs (`NumPy`, `Keras`, and `TensorFlow` versions are available) are in the `forward_models/python_models` directory. The `C++` implementation is in the `forward_models/cpp_models` directory. The `MPE_examples` directory gives some basic examples of traditionally trained neural networks, implemented in `Keras` and `TensorFlow`.

# User guide

Note we do not currently have a user guide, but the code is well documented. If you are interested in learning more about running the code, please don't hesitate to send me an email at: `kj316@cam.ac.uk`.

# Interesting datasets
- Kaggle: https://www.kaggle.com/
- driven data (similar to kaggle): https://www.drivendata.org/competitions/
- innocentive (ditto): https://www.innocentive.com/
- tunedit (ditto): http://tunedit.org/challenges/
- openml (ditto): www.openml.org
- UCI database of datasets, includes counts of number of records and number of features, as well as papers which cite the datasets: https://archive.ics.uci.edu/ml/datasets.html?sort=nameUp&view=list

# summaries of other large scale works on bnns

- Mackay uses Gaussian approximation to get solutions to BNNs, approximates posterior centred around most probable parameter values which are found via optimisation. Error is estimated from Hessian. This approximation assumes posterior is unimodal, and breaks down when number of parameters approaches numbr of datapoints. Uses this analytic approximation to calculate the evidence. Picks value of hyperparameters which maximise evidence, which equivalently maximise the posterior of the hyperparameters given the data, for a uniform prior on the hyperparameters. Thus essentially assumes that marginalising over hyperparameters and taking their maximum are equal, i.e. the hyperparam posterior is also Gaussian. then looks at evidence values given these best hyperparameters and training set error to evaluate models. Uses some approximation of the evidence maximisation to update the hyperparameter.

- finds when a rubbish model used, evidence and test error not as correlated as when good model is used. Further, the evidence is low in some cases where test error is good. Uses this to deduce structure of model is wrong. Also sees Occam's hill.

- Likelihood variance is fixed. Initially one hyperparam used for all weights and biases. Found evidence and generalisation don’t correlate well. MacKay argues this is because the scales of the inputs, outputs and hidden layers are not the same, so one cannot expect scaling the weights by the same amount to work well. So then tried one hyperparam for hidden unit weights, one for hidden unit biases, then one for output weights and biases. This gave higher evidence, higher test set performance, and stronger correlation between the two

- Neal uses HMC to sample the BNN parameters, and Gibbs sampling to sample the hyperparameters. n.b. HMC requires gradient information, so can't be used to sample hyperparameters directly (to my knowledge). Also, HMC in essence has characteristics similar to common optimisation methods which use 'momentum' and 'velocity'.

- Neal also introduces concept of using Gaussian processes to introduce a prior over functions, which tells us what nn predicts mapping function to be without any data.

- From Neal it seems that sampling hyperparameters seems rather necessary to justify allowing NN to be arbitrarily big- if complex model is not needed, hyperparameters will 'quash' nodes which aren't important, according to the hyperparameter values assigned by the data during the training, and 'upweight' important nodes. Also avoids cross validation step.

- Uses stochastic/mini-batch methods.

- Neal's result (with same simple model as Mackay) on test data is similar to the Mackay best evidence model's results, but not as good as his best test error model results. Performance didn't necessarily get worse with larger networks for BNNs, but did for MAP estimates (though don't think this treated hyperparameters as stochastic).

- n.b. hyperparams in first layer indicate which inputs to network are important. using it generalises to test data better, as irrelevant attributes fit to noise in train. Furthermore, Neal scales hyperprior (gamma parameter w, which is mean precision) by number of units in previous layer i.e. for layer i w_i -> w_i * H_{i-1} for i >= 2 (note he doesn't scale first hidden layer). Note that he does not use this scaling on the biases, in particular, for hidden layers, biases are given standard hyperprior, and for output layer the biases aren't given a stochastic variance at all (instead they are usually fixed to a Gaussian with unit variance).

- larger network is, more uncertain it is to out of training distribution data.

- for bh, BNN does much better on test error than traditional (though I don't think this uses cross validation in traditional sense).

- Freitas uses reversible jump MCMC to sample neural network systems. reversible jump MCMC is necessary when number of parameters changes. This is the case here, as the number of radial basis functions (neurons) is allowed to vary in the analysis, resulting in a varying number of model parameters/hyperparameters throughout the sampling. Gives posteriors on number of functions, as well as the usual param/hyperparams ones.

- Also uses SMC to train NNs where data arrives one at a time. Idea is to model joint distribution of model parameters at each timestep, and appears to do a better job of predicting it with more time/data.

- Also does model selection, using posterior over number of basis functions. Can do this in sequential context as well. 

- Finds reversible jump MCMC does as well as Mackay and Neal, and better than expectation maximisation algorithm (which is similar/equivalent to variational inference), but is slower than EM algo.

- Gal provides the genius insight that stochastic draws from the distribution over neural networks can be done using traditional methods. Usually if using dropout regularisation, one disables the dropout once training is finished. Gal shows that using dropout during model deployment is equivalent to using variational inference to get a probabilistic model output. The parameters of the variational inference problem are determined by the dropout properties I believe. The higher the dropout probability, the stronger the prior on the inference problem.

- This essentially means a Bayesian approach can be used even for high dimensional problems, the training time is the same as that of maximisation methods, and during deployment, one is only limited by how many samples from the posterior one wants.

- Gal finds that this method exceeds traditional variational inference methods both in terms of speed and test set performance for most tasks, with the only doubts occurring in some CNNs. He also finds it outperforms traditional methods in terms of test set performance, with the added bonus that one gets an uncertainty estimate. The method however cannot give evidence estimates.

# background reading on BNNs

- theses which focus on bnns and their applications, to draw inspiration from: http://mlg.eng.cam.ac.uk/yarin/blog_2248.html#thesis (Gal, 2016), www.cs.ubc.ca/~nando/papers/thesis.pdf (Freitas, 1999), http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.446.9306&rep=rep1&type=pdf (Neal, 1995), http://www.inference.org.uk/mackay/PhD.html (Mackay, 1991) 

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

- papers on restricted Boltzmann machines: www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf, https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
