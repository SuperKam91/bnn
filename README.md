# bnn
Training Bayesian neural networks using Markov chain Monte Carlo techniques.

Qualitative document giving an overview of BNNs and the features of this package:
https://www.overleaf.com/5218885132gxxndzvrfhfn
  
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

# 21cm stuff

## data

- Seems that for 7 inputs (cosmo parameters), 21cm signal is given as a function of redshift, over 136 redshift bins. i.e. input to nn is m x 7, output is m x 136. For training, m = 21562, for testing, m = 2073. 

## first thoughts on modelling process

- From the data I initially thought that the objective was to predict the 21cm signal as a function of redshift, and so came up with the following ideas (turns out I was completely wrong):

- For 7 inputs (cosmo parameters), predict 21cm signal as a function of redshift, over the 136 redshift bins. i.e. input to nn is m x 7, output is m x 136. However, this method would require a separate interpolation step for predictions at redshifts other than the discrete values used in training.

- Could also formulate the problem differently. Could instead consider 136 separate nns each having 8 inputs (the 7 cosmo params and one redshift value), each having one output (the 21cm signal at that redshift), so that for each nn the input is m x 8 and the output is m x 1. However since each nn corresponding to the signal at different z is trained independently, think this would lose some information compared with above approach. This is because one would be sampling independently from 136 likelihoods rather than a joint likelihood over the full signal, which would have a profound effect in deep neural networks where a hidden layer parameter influences multiple outputs. Method would also require interpolation.

- I think the best approach would be to treat z as an input parameter, and treat each of the 136 outputs as separate records of the data corresponding to one output (the 21cm signal). In this case the input to the nn is (m * 136) x 8 and the output is (m * 136) x 1. Technically this would give a nn whose output is continuous, but since the redshift values used for training only take a small number (136) of different values compared with the size of the training data, I'm not sure how well the function will generalise to out of training distribution redshift values.

## thoughts on modelling after looking at paper

- Turns out objective is to predict 21cm signal as a function of frequency (which isn't given?), and to do this a number of steps are taken, involving several regression and classification nns. The main premise is that a nn is used to learn the coefficients of a PCA (coefficients of each basis function) which spans the dataset, as well as key points of the signal.

- Basically, uses nn with inputs and outputs of parameters I don't have, to make classifications and regressions, to make decisions on how many PCA components to use, the values of their coefficients, and the value of the emission at key astrophysical points.

- For nns, use one hidden layer with 40 neurons, using the Levenberg–Marquardt algorithm to minimise the cost function, which I believe uses second order gradient information (and isn't implemented in tf as it only has first order methods).

- The bagged trees algorithm is used for the classification problem. This algorithm ﬁts many decision trees, each time using a diﬀerent subset of the training set, and the decision is made by voting. After optimisation they choose to use bagged trees with 30 tree learners and the tree size was chosen using 5-fold cross-validation. 

- To me it seems like machine learning is a relatively small part of this work. Much more time and effort is focused on the understanding of the underlying physics, and constructiing this pipeline of analyses than the actual training of regression/classification functions.

## further thoughts

- They calculate error on each time series for given parameter set (~1700 of these for testing data), and plot histogram of errors associated with each of these sets.

# other stuff

- Compare performance of different architectures, different granularity of stochastic hyperparameters on Boston housing dataset in terms of Evidence and test set errors.
- Think of L0 regularisation in terms of a prior.
- Look at treating likelihood variance as stochastic.
- See how 'infinitely' long ResNets perform c.f. universal approximation using ResNets paper.
- Start thinking about how to adapt pipeline so that number of layers, number of neurons and type of activation functions can be treated as stochastic hyperparameters.
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

- n.b. hyperparams in first layer indicate which inputs to network are important. using it generalises to test data better, as irrelevant attributes fit to noise in train.

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

- In general tf models can't be used with sklearn's grid/random search methods, but apparently it can be used with: https://ray.readthedocs.io/en/latest/tune.html. tf required to reach most granularity in hyperparameter tuning e.g. different regularisation constants for each input to layer (c.f. input_size granularity in bnns).

- Turns out you can also use keras models with sklearn's gridsearch (and presumably randomised search), see: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/. This allows one to be more granular with hyperparameters, such as regularisation constant for each layer, number of nodes per layer, number of layers, different activations for each layer, dropout regularisation, etc. Requires the use of keras->sklearn wrappers, see: https://keras.io/scikit-learn-api/.
Note to change ll variance, need to use a hack: define a higher order loss function which takes variance as argument, and returns loss function with that variance which can then be passed to model.compile. On each iteration, new higher order function will need to be called with the new variance value, and corresponding loss function passed to model.compile.

- Randomised search can be thought of probabilistically as the same as gridsearch, but with non-uniform priors in general.

- In sklearn you can also use a randomised search over network hyperparameters. Where instead of exhausting the hyperparameter grid, you sample from it according to probability distribution(s) governing each dimension of the grid.

- Note grid search is sort of equivalent to assigning uniform priors to the hyperparameters, and then maximising the conditional posterior of the model parameters (conditional on the hyperparameters) on the training data. The maximisation of the conditional hyperparameters posterior (conditioned on the model parameters, which are different for each hyperparam set) is then obtained by  maximising over the cross validation data (or the best average in the case of k-fold). Not sure it's possible to formulate this mathematically, but I guess it can be interpreted as an approximation to the full posterior.

- In sklearn you can perform a gridsearch over network hyperparameters of the sklearn nn models (reg constant of prior, number of layers and sizes, optimiser, optimisation hyperparameters, etc.). For regression problems this uses k-fold cross validation (for each run, partition training data into k equally sized sets, train network on k-1 of these, and use remaining set as cross validation set. Repeat this k times with the cross vaidation set being one of the different k sets each time, then average over the k evaluation metric scores to get the final metric value. The 'best' set of hyperparameters can then be used to train a model on the whole (all k-folds) of the training data, giving a final model). For classification it uses stratified k-fold (same as vanilla version but has roughly same proportion of each class in each fold).  

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
