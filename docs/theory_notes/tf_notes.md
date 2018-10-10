- tensorflow calculates derivatives using reverse mode automatic differentiation.

- can't find any information on whether it whitens data before calculating derivative, just know that it is a standard pre-processing step in ml, and batch normalisation layers within NN architectures are common.

- automatic differentiation calculates derivatives by breaking down functions into primitive functions and applies the chain rule.

- forward mode automatic differentiation starts at the independent value and calculates the derivatives back through the chain until it reaches the dependent variable (adv: intermediate values don't need to be stored, disadv: multiple sweeps forward need to be done if dependent on more than one independent variable).

- reverse mode differentiation first calculates the derivative of the dependent variable w.r.t. the variable next in the chain and propagates through to the independent variable (adv: only has to done one sweep regardless of number of independent variables, disadv: have to store intermediate values). reverse-mode makes sense in context of NNs as it naturally fits backward propagation flow from J -> inputs.

- one epoch is one forward pass and one backward pass of all the training examples.

- batch size is the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

- number of iterations is the number of passes (forward + backward propagation), each pass using [batch size] number of examples.

- dropout is a regularisation technique, only used during training. for a given layer and 'keep' probability, you keep each node in a pass with probability keep probability, and ignore it with 1 - keep probability. for corresponding activation, usually divide by keep probability to keep expected value of activation constant (helps stabilise activation values), as on average, activation function will be reduced by a factor of keep probability due to nodes ignored. Intuition is that network can't rely on any one node, so spreads weights more evenly.

- momentum optimisation technique introduces a "velocity" into the gradient descent methodology to help guide the solution out of local optima. The velocity is initialised at zero and the velocity and position are updated as: v = mu * v - learning_rate * dx; x += v where dx is derivative of the cost function w.r.t parameter x. 1 - mu is equivalent to a frictional term, and is decreased so that the solution converges. v can be interpreted as some linear combination of previous gradients (with latest having most weight)

- nesterov momentum updates the position with the "velocity" first and then calculates the velocity at this new point, and uses this to update the original position. We know that the momentum term alone (i.e. ignoring the second term with the gradient) is about to nudge the parameter vector by mu * v. Therefore, if we are about to compute the gradient, we can treat the future approximate position x + mu * v as a “lookahead” - this is a point in the vicinity of where we are soon going to end up. Hence, it makes sense to compute the gradient at x + mu * v instead of at the “old/stale” position x. The update process is: x_temp += mu * v; v = mu * v - learning_rate * dx_temp; x += v.

- second order methods such as Newton's method which rely on the Hessian matrix are seldom used in deep learning as they are too computationally expensive.

- adagrad divides each component of dx by the sum of the previous dx values in that dimension i.e. x += - learning_rate * dx / sqrt(sum_dx_prev^2). The effect of this is that dimensions that have had large changes in the past are less likely to be changed much compared to ones that have only experienced small moves thus far.

- rmsprop adds to adagrad by adding a decay rate to the sum of square dx_prevs: sum_dx_prev^2 = decay_rate * sum_dx_prev^2 + (1 - decay rate) * dx^2. This decreases the rate at which sum_dx_prev^2 increases, meaning the changes in don't shrink as quickly.

- Adam is similar to rmsprop which includes a moving average of both the gradient and sum_dx_prev^2. The simplified version is:m = beta1*m + (1+beta1)*dx; v = beta2*v + (1-beta2)*(dx^2); x += - learning_rate * m / (np.sqrt(v) + eps) where beta1 and beta2 are decay rates. The idea of moving averages is to make the estimate (m and v) less noisy. The unsimplified version removes biases on m and v associated with the fact that they are initialised to zero. Adam generally gives best results.

