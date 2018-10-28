import numpy as np
import scipy.special


class UniformPrior:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a + (self.b-self.a) * x


class GaussianPrior:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return self.mu + self.sigma * numpy.sqrt(2) * scipy.special.erfinv(2*x-1)


class LogUniformPrior(UniformPrior):
    def __call__(self, x):
        return self.a * (self.b/self.a) ** x


def forced_indentifiability_transform(x):
    N = len(x)
    t = np.zeros(N)
    t[N-1] = x[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = x[n]**(1./(n+1)) * t[n+1]
    return t


class SortedUniformPrior(UniformPrior):
    def __call__(self, x):
        t = forced_indentifiability_transform(x)
        return UniformPrior.__call__(self, t)


class LogSortedUniformPrior(LogUniformPrior):
    def __call__(self, x):
        t = forced_indentifiability_transform(x)
        return LogUniformPrior.__call__(self, t)

p = np.array([0.1, 0.9, 0.2, 0.8])
print SortedUniformPrior(-1., 1.)(p)