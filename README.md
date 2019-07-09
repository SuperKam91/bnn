# bnn
Training Bayesian neural networks using Markov chain Monte Carlo techniques.

Qualitative document giving an overview of BNNs and the features of this package:
https://www.overleaf.com/5218885132gxxndzvrfhfn

Currently runs with PolyChordLite version 1.15. e.g. checkout 316effd815b2da5bafa66cfd0388a7c601eaa21d
  
# Interesting datasets
- gresearch financial forecasting challenge: https://financialforecasting.gresearch.co.uk/
- Kaggle: https://www.kaggle.com/
- driven data (similar to kaggle): https://www.drivendata.org/competitions/
- innocentive (ditto): https://www.innocentive.com/
- tunedit (ditto): http://tunedit.org/challenges/
- openml (ditto): www.openml.org
- UCI database of datasets, includes counts of number of records and number of features, as well as papers which cite the datasets: https://archive.ics.uci.edu/ml/datasets.html?sort=nameUp&view=list


# anyvision stuff

- Bayesian GANs have been shown to give superior performance due to their tendency not to get stuck in local optima, which occurs in GAN training when the generative model finds an 'easy' solution which fools the discriminator.

- ResNets with one neuron per layer have been shown to satisfy their own universal approximation theorem, meaning they can approximate a function (with certain restrictions) to an arbitrary level of precision, as the length of the network approaches infinity. Since it is of Resnet architecture, each layer should be the same size dimension as the input layer. 

However looking at the Wikipedia example (a_2 = g(z_1 + a_0)), this can arise in two ways. if a_0 has size 1 x n, a_1 can be forced to have the same size if a_1 = w_1 * a_0 + b_1 with w_1 having size 1 x 1, or it can be given by a_1 = a_0 * w_1 + b_1 with w_1 having size n x n. Thus odd indexed layers have dimensionality 1 + n or n * n + n and the even layers have no learned parameters.

In the universal approximation paper, the architecture and dimensions seem somewhat different, and describes it in terms of 'blocks'. It appears that a_2 is given by a_2 = w_2 * g(a_0 * w_1 + b_1) + a_0 with w_1 having size n x 1 and w_2 1 x n. It is interesting that the second layer doesn't have a bias, or an activation. I assume the output of each block is used as the input to the next. With this architecture each block corresponds to a one neuron layer followed by an n-wide layer. Thus each block has 2*n + 1 parameters, n + 1 for the first layer, and n for the second.

The coursera version is the same as the universal approximation paper, but the second layer of each block contains a bias and an activation over the whole input, i.e. a_2 = g(w_2 * g(a_0 * w_1 + b_1) + b_2 + a_0). Thus each block has 3*n + 1 parameters, n + 1 for the first layer, and 2n for the second (n for the weights and n for the biases).

 I believe this is the exact version introduced in the original ResNet paper. The final layer is a linear (potentially multi-neuron) layer. 
In paper, ResNet does better than wider NN for simple example (both are tested up to 5 layers).

- Look at hyperspherical energy paper

# main work

- Run mlp bnns first with no stochastics, then with different granularities of prior and stochastic var. At each stage, repeat for different sized nns. Idea is to see if evidence is correlated with test set performance as the nns change in their complexity. NOTE FOR RUNS BH_50_R_MLP_2, BH_50_R_MLP_4, AND BH_50_R_MLP_8, THINK I ACCIDENTALLY PUT SORTED GAUSSIAN PRIOR ON OUTPUT LAYER INTERCEPT INSTEAD OF VANILLA GAUSSIAN. SHOULDN'T HAVE ANY EFFECT AS DEGEN DEPENDENCE LENGTH IS 1.
- Can then piece together these runs to see how evidence/predictions change as a function of the hyperparameters dictating the neural network size, and activation type. Nice divide here as these are discrete hyperparameters.
- Will be good to see how test performance varies as a function of these things. May be good to also compare with traditional equivalents, where they use randomised search/Bayes optimisation to select model hyperparams. However, getting same granularity on hyperparameters will be very difficult in some cases (e.g. input_size granularity on stochastic prior hyperparams). 
- Also want to take advantage of polychord’s no derivatives requirement. May be the case that nns which are usually associated with numerical difficulties (e.g. very deep nns or ones with more complex activations) actually perform very well when trained with PolyChord, as these issues are no longer relevant.
- Furthermore, not needing derivatives is relevant for the training some of the stochastic hyperparams (number of layers, nodes, activation types). As with these params, even using hierarchical Bayes one cannot minimise the objective function.
- On that note, could look at traditional performance using hierarchical Bayes on stochastic var and prior hyperparams as I don’t think this has ever been done before. However, don’t think I will do this

## nn runs

- The aim of Bayesian neural networks is to consider the whole space of networks modelled probability by the posterior distribution on the network parameters, conditioned on the training data. With this posterior distribution, one can make predictions of the outputs of the network, and quantify the model certainty on these estimates. 

- The predictions from a BNN are often calculated by marginalising over the network parameters, rather than just using their single maximum likelihood/posterior point, as is done in traditional methods.

- While in traditional regression NN training, the average squared loss is commonly chosen as the objective function to minimise (plus some regularisation on the network parameters), in the Bayesian framework, a Gaussian distribution is commonly used to describe the likelihood of the data, given the network parameters, and the priors on these network parameters are also often chose to be Gaussians. Furthermore one can use hierarchical Bayes to treat the likelihood and prior variances as random variables (regularisation factors in traditional methods). In the work here, this is referred to as the stochastic hyperparameter approach (see below). Note this method is by no means new in the context of BNNs, despite the fact that it is seldom used in traditional NN training methods.

- Various architectures and setups are trained to see how the test set error correlates with the Bayesian evidence. For each setup, a BNN is trained on 10 different randomisations of the train/test set split, each of which gives a value for the test set error and Bayesian evidence (and their errors). The total size of the BH dataset is 506, with 13 inputs and one output. a 50-50 split was used for the training and test data.

- The following architectures were considered: (0), (2), (4), (8), (2,2), (4,4), (2,2,2), (4,4,4), (2,2,2,2), (4,4,4,4).

- For each architecture, the following nn types were considered: tanh activation no stochastic hyperparameters (fixed variances on likelihoods/priors), relu activation no stochastic hyperparameters, tanh activation single stochastic prior hyperparameter (one hyperparameter for variance of all priors), tanh activation layer prior hyperparameters (one hyperparameter for weights in each hidden/output layer, one hyperparameter for biases in each hidden/output layer), tanh activation input_size stochastic prior hyperparameters (for given layer, one hyperparameter for each input to that layer, which is shared among all weights in the layer multiplying that same input. Also one hyperparameter for biases per hidden/output layer). For all stochastic hyperparameter runs one hyperparameter is used for the variance of the likelihood.

- Note for non-stochastic hyperparameter systems, likelihood variance is fixed to unity, all priors are zero mean Gaussians with unit variance. For stochastic hyperparameter systems, the prior hyperparameters control variance of priors of NN weights/biases (i.e. regularisation factors). The prior on these hyperparameters is in the form of a Gamma distribution on the precision of the prior distribution (1 / variance). The likelihood variance has a hyperprior assigned to it in the same way. For single prior hyperparameter granularity and the likelihood hyperparameter, the gamma prior has hyperparameters a = 1, b = 1. For layer and input_size granularities on the stochastic prior hyperparameters, the first hidden layer weights and all biases have Gamma priors parameterised by a = 1, b = 1. The other layer weights have a Gamma prior with a = 1, b = 1 / number of nodes in previous layer. 

- For combined runs, combine various different BNNs which were trained independently, using a uniform prior over the categorical variables describing these combinations, and also weighted by their relative Z values. Could have trained these networks in one run, treating categorical variable as sampling parameter (and arrive at the same result, theoretically). However, it seemed natural to treat the discrete categorical parameters as partitions for the individual runs, to better manage computational resources.

- Looked at combining different architectures with same activations (and thus different sizes and stochastic hyperparams), same stochastic hyperparams (and thus different sizes), and similar sizes in some sense (e.g. number of nodes per layer, or number of layers, and thus different stochastic hyperparams), as well as combinations of these combined networks (their supersets).

- The PolyChord algorithm was run with 1000 livepoints and n_repeats set to 5 * the dimensionality of the parameter space.

## results

- Overall looking at the (averaged) full results of Z vs test loss there are three modes apparent, corresponding to no stochastic hyperparameter tanh activation bnns, no stochastic hyperparameter ReLU activation bnns, and stochastic hyperparameter tanh activation bnns.

- For a given BNN, there is no correlation between Z and test loss over different data (randomisations).

- Looking at non-stochastic hyperparameter runs and a given activation, no trend between evidence and test set performance was seen over different sized neural networks. However, surprisingly, ReLU activation consistently performs better than tanh (with no stochastic hyperparameters, ReLU activation with stochastic hyperparams were not tested) over all networks in terms of test set performance. This is also reflected in the higher associated evidence values

- With the single stochastic hyperparameter networks, start to see a (negative) correlation between Bayesian evidence and test set performance. Also, start to see a remarkable symmetry between the log evidence as a function of nn dimensionality and the test set performance as a function of the same parameter. Also, start to see a peak in the Bayesian evidence, which is fast to increase (with nn dimension), but relatively slow to decline. This is a well-known trend in model selection. The corresponding dip in test set performance is less well-pronounced, but still arguably there (or alternatively, could be interpreted as a plateau).

- For the layer stochastic hyperparameter networks, more of a correlation between test set performance and Bayesian evidence is present. Similar to the single stochastic hyperparameter case, the ln(Z)-nn_dim and test set performance-nn_dim symmetry also appears. The correspoding peaks and troughs are also there, but are less clear-cut (more plateau-like).

- Although, surprisingly, the input_size stochastic hyperparameter networks consistently don't perform as well as the layer systems (of the same network size). Since this system is more granular than the layer systems, it should be able to perform at least as well as it. Thus, one can only attribute this to the (more) complex parameter space not being explored as well, or, the system overfitting to the training data. Nevertheless, the input_size results show a good correlation between test performance and evidence, a strong symmetry in evidence and test performance (see points above), and arguably, the most well-formed peaks/troughs in the same graphs.

- Looking at all the stochastic hyperparameter systems, the correlation and symmetries remain, but the peaks/troughs are much less apparent (more of a plateau).

- Now turning attention to all the 2-node architectures, the correlation still exists, with clear modes corresponding to tanh no stochastic hyperparams, ReLU no stochastic hyperparams, and the tanh stochastic hyperparam systems. The symmetries are still present, but the peaks/troughs/plateaus (if they exist) are convoluted. 4- and 8-node architectures show similar stories.

- Looking at systems grouped together by how many hidden layers they contain (1, 2, 3 or 4), the correlations exist, but there seems to be mainly two modes of separation: non-stochastic hyperparams vs stochastic hyperparams. For one layer, all but the (8) systems show good peaks/troughs. Similar can be said for the 2, 3, and 4 layer systems, but the peak/trough relation is weaker.

- For the combined network results, overall data shows same three modes as individual runs

- For all of the combinations considered, Z-test set performance trends were very similar to the results of the networks which these combined runs comprised of. Combined log evidences often laid in middle of values which individual ones had (makes sense of course). Test set performance was also very similar on average. Note however the lowest test loss was a combined run (all networks combined), but the average test loss of the combined networks was higher than the individual runs. Latter may be due to a lot of models having significantly larger evidences than others, making such a model averaging (by Z) strongly dependent on these particular models. 

- Also trained same network sizes (with tanh activations) using traditional methods in the most basic way (no regularisation, no hyperparameter tuning), with 1000 epochs on the same train/test data splits. 6 of the  MLE test set estimates estimates were inferior to the no stochastic hyperparameter BNN estimates, while 4 were superior. All 10 BNNs performed better than their MLE equivalents when stochastic hyperparameters were used. 

# other stuff

- Think of L0 regularisation in terms of a prior.
- See how 'infinitely' long ResNets perform c.f. universal approximation using ResNets paper.
- Maybe also think about how at least some parts of the training can be done traditionally (via optimisation) and as a bonus, how one can obtain uncertainty and evidence estimates from this subspace by using the Hessian.
- Look at typical dimensionality of sequential/recurrent nns.
- Consider looking at Edward results (google's Bayesian nn package which uses variational inference approach) and compare with Polychord.

# discarded stuff

- Will forget about FIFA dataset runs for now, and concentrate on Boston housing

# summaries of other large scale works on bnns

- Mackay uses Gaussian approximation to get solutions to BNNs. Picks value of hyperparameters using a form of evidence (using analytical solution I believe), then looks at 'usual' evidence values and training set error to evaluate models. Think he finds that test set performance and evidence are positively correlated, and that evidence as a function of size of neural network increases then dips at overly complex models.

- finds when a rubbish model used, evidence and test error not as correlated as when good model is used. Further, the evidence is low in some cases where test error is good. Uses this to deduce structure of model is wrong. Bayesian does better than traditional

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

# background reading

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

- papers on restricted Boltzmann machines: www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf, https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

# thoughts scratchpad

- For the runs with stochastic hyperparameters I have made a bit of an adaption. According to Neal, the biases should never be scaled (unlike the weights). Thus for the layer and input_size granularity cases, the biases are given their own hyperpriors. For consistency I also give the biases in the first hidden layer their own separate hyperpriors (as sharing them with weights in first layer doesn't seem fair, if in other layers, where biases aren't scaled but weights are, they get their own separate hyperprior).

- Posts on width vs length of neural networks: 
https://stats.stackexchange.com/questions/222883/why-are-neural-networks-becoming-deeper-but-not-wider;
https://www.quora.com/Why-are-neural-networks-becoming-deeper-more-layers-but-not-wider-more-nodes-per-layer;
https://stats.stackexchange.com/questions/214360/what-are-the-effects-of-depth-and-width-in-deep-neural-networks;
https://stats.stackexchange.com/questions/182734/what-is-the-difference-between-a-neural-network-and-a-deep-neural-network-and-w;
basic idea is that wider networks develop more lower level features/ ones 'similar' to inputs, while deeper networks calculate more complex features more abstract from the inputs. Arguments for depth are that more complex functions of the input can be learned and thus the same 'predictive power' may be obtained with fewer nodes, and more abstract features may generalise to out-of-sample data better. Arguments for width are that it is more computationally efficient as more calculations can be done in parallel, less problems with numerical instabilities such as vanishing and exploding gradients.
Papers on the topic:
https://arxiv.org/abs/1312.6184;
https://arxiv.org/abs/1512.03385;
https://arxiv.org/pdf/1605.07146.pdf

- Look at number of constrained dimensions by calculating from likelihood values in chains (c.f. Will's new paper).

- Calculate likelihood value associated with traditional training (Keras) weights. Will be interesting to see how these compare with likelihood values of nested sampling

- L0 norm is a much desired regularisation type in traditional methods, but is difficult to compute. Wonder what prior this corresponds to.

- As these models become increasingly complex as we let the data decide more hyperparameters (prior variance, likelihood variance, number of layers, number of neurons per layer, type of activation functions, network parameters), may be worth training some of these (i.e. the differentiable ones) using traditional (optimisation) methods. One can still get a measure of uncertainty and an estimate of the evidence on this subspace by looking at the Hessian.

- Likewise neural network size can be either judged by evidence (and see if it correlates with test set error), or chosen by data (by treating as a stochastic hyperparameter) and marginalised over.

- Activation functions can be treated as a stochastic parameter: assign a categorical variable to the type used. Then can marginalise over this variable when making prediction, so we are essentially considering an ensemble of networks (with different activations and weights) to make the prediction. Also look at evidence/test error (and compare with models where activation isn't varied), see if switching activations worth it.

- Same can be said with network size, as different architectures can lead to different numerical instabilities in the gradients.

- Would be interesting to see if bnns show activation which fits the data particularly well (by looking at evidence / test error), where gradients aren't important.

- n.b. in Neal, scaling of hyperprior params (by number of units in that layer) is only applicable to hidden layers, and in particular, not to the biases.

- Look at success of global average pooling layers (GAP). Premise is for an input layer of size (num_batches, height, width, depth), take average value over height and width to return a tensor with shape (num_batches, 1, 1, depth). Useful for e.g. drastically reducing size of any subsequent fully connected layers. See:
https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/

- Transfer learning is a potential opportunity: train all but final layer using MLE methods (or use pre-trained weights) to e.g. learn featuring generating nodes. Train final layer(s) with BNN to learn accurate representation of non-linear combination of these learned features, and some uncertainty. Problem is, previous layers cannot be too big, or even training the final layers will not be possible. Unfortunately this will often be the case with CNNs, as the number of channels usually increases with depth into the NN (as the CNN learns a larger number of intricate features as one delves into the NN).

- An alternative to the above is using MLE/MPE methods to train an (variational) autoencoder to reduce the dimensions of the input, and then feed this to a BNN. Downside of this is you will lose some information in the encoding of the input, which may outweight the gains of treating the problem in a Bayesian framework

- Seems that there are applications in ML in areas such as medicine, where number of training data is small. Thus one is forced to use small networks to prevent overfitting. See bullet below.

- https://reader.elsevier.com/reader/sd/pii/S0933365716301749?token=D3ADCF7EC51CE6EBC44A540012EAAFB1F5D8CFC8BA12B847B380A83E838D1F4444E61168F6FA06D9F43A5D794D04504D describes a regression problem with five input features and 35 records. Use a one layer NN (~40 parameters), but train over 20000 NNs (with different parameter initialisations, and different randomisations of data for stochastic gradient descent), and pick NN with best performance on cross-validation set, for their final evaluation. Use RMSE and R^2 regression coefficient to evaluate performance, and get values RMSE = 0.85 and R^2 = 0.983 for final evaluation. I believe they run the algorithms for ~30 epochs.

- seems BNNs are useful in approximating Q(s,a) function in RL- rather than approximating maximum of Q(s,a) for values of states and actions, better to have a probability distribution over each Q(s,a), so a can be sampled. Sample from each Q(s,a) for a given state, and choose a corresponding to highest Q value. The stochasticity introduced in sampling a supposedly helps the exploration of the algorithm. see: https://www.coursera.org/learn/practical-rl/lecture/okvvc/thompson-sampling

- look into applications of bnns to deep reinforcement learning- believe they are useful as in rl aren't always looking for optimal value function approximation, having stochastic output helps with exploration of algorithm.

- for pure nn applications (i.e. regression and classification) in general if a lot of data is available, can build an almost arbitrarily large network these days (thanks to gpus) to perform well, regardless of the complexity of the task. Thus we might want to focus on examples where data is scarce. In these cases, building complex neural networks could lead to overfitting, thus simple networks which perform well (hopefully bnn) should be optimal

- I also suspect that bnns are more useful for regression than classification. In the latter case, MLE/MPE already gives a probability for each class output, thus a bnn is just giving a 'probability distribution of probabilities', which in isolation isn't very satisfying (though the statistics one can calculate from these distributions are still interesting). On the other hand in regression, having a probability distribution over some output variable will almost always be useful

- will probably need to ignore all explicit regularisation techniques (i.e l^n norm regularisation) as polychord handles prior separately from likelihood/cost function. OR could handle prior in cost function using regularisation, then assign uniform priors in polychord

- distributed tf uses master & slave and mapreduce paradigms. need to see if we can use distributed tf on cambridge hpc

- tf and keras can use gpus almost automatically. very little intervention needed. need to see if we can use tf-gpu on cambridge hpc gpu systems

- need to think how batches will be handled with polychord (opposed to passing all data at once)

- one idea is to use tf/keras optimisation a little bit to get a good 'initial guess' of the network parameters. pass these to polychord and it can use these as the initial livepoints. note however these initial livepoints would have to be independent. subsequent samples will be at likelihoods higher than the initial set, and so should be able to concentrate on sampling the peak of the posterior. avoids wasting time sampling 'unreasonable' network parameters. another problem with this is the initial livepoints aren't sampled according to the prior, unless we implement the prior (regularise) in the optimisation exploration (then can include prior in ll func to pc with uniform priors or treat priors separately as normal)
