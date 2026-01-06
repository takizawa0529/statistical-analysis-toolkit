from abc import ABC, abstractmethod

import numpy as np

class BaseDistributionMoments(ABC):
    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def var(self):
        pass


class DiscreteUniformMoments(BaseDistributionMoments):
    def __init__(self, lower:float, upper:float):
        self.lower = lower
        self.upper = upper
        if self.lower > self.upper:
            raise ValueError("upper must be greater than lower.")

    def mean(self):
        return (self.upper + self.lower)/2

    def var(self):
        return ((self.upper - self.lower + 1)**2 - 1)/12


class BernoulliMoments(BaseDistributionMoments):
    def __init__(self, p: float):
        self.p = p
        if not 0 <= p <= 1:
            raise ValueError("p must be 0 <= and <= 1.")
    
    def mean(self):
        return self.p

    def var(self):
        return self.p * (1-self.p)


class BinomialMoments(BaseDistributionMoments):
    def __init__(self, n: int, p: float):
        self.n = n
        self.p = p
        if not isinstance(n, int):
            raise ValueError("n must be int.")
        if not 0 <= p <= 1:
            raise ValueError("p must be 0 <= and <= 1.")
    
    def mean(self):
        return self.n * self.p

    def var(self):
        return self.n * self.p*(1-self.p)

# HyperGeometricDistribution
class HyperGeoMoments(BaseDistributionMoments):
    def __init__(self, N: int, M: int, n: int):
        self.N = N
        self.M = M
        self.n = n
        if not isinstance(N, int):
            raise ValueError("N must be int.")
        if not isinstance(M, int):
            raise ValueError("M must be int.")
        if not isinstance(n, int):
            raise ValueError("n must be int.")
        if (n>N or n<=0) or (M>N or M<=0):
            raise ValueError("n and M must be 0 < and <= N")
        if (N<=0):
            raise ValueError("N must be greater than 0.")

    def mean(self):
        return self.n * self.N/self.M

    def var(self):
        return self.n * self.M/self.N * (1 - self.M/self.N) * (self.N-self.n)/(self.N-1)

# Poisson Distribution
class PoissonMoments(BaseDistributionMoments):
    def __init__(self, L):
        self.L = L
        if L<=0:
            raise ValueError("L must be greater than 0")
        
    def mean(self):
        return 1/self.L

    def var(self):
        return 1/self.L


# Grometric Distribution
class GeometricMoments(BaseDistributionMoments):
    def __init__(self, p: float):
        self.p = p
        if not isinstance(p, float):
            raise ValueError("p must be float or 0, 1.")
        if not 0 <= p <= 1:
            raise ValueError("p must be 0 <= and <= 1.")

    def mean(self):
        return (1-self.p)/self.p

    def var(self):
        return (1-self.p)/self.p**2

# Negative Binomial Dostribution
class NBMoments(BaseDistributionMoments):
    def __init__(self, r: int, p: float):
        self.r = r
        self.p = p
        if not isinstance(r, int):
            raise ValueError("r must be int.")
        if not 0 <= p <= 1:
            raise ValueError("p must be 0 <= and <= 1.")
        if r<0:
            raise ValueError("r must be greater than 0.")

    def mean(self):
        return self.r * (1-self.p)/self.p

    def var(self):
        return self.r * (1-self.p)/self.p**2

# Continuous Uniform distribution
class ContinuousUniformMoments(BaseDistributionMoments):
    def __init__(self, lower:float, upper:float):
        self.lower = lower
        self.upper = upper
        if self.lower >= self.upper:
            raise ValueError("upper must be greater than lower.")

    def mean(self):
        return (self.upper + self.lower)/2

    def var(self):
        return (self.upper - self.lower)**2/12


# Normal Distribution
class NormalMoments(BaseDistributionMoments):
    def __init__(self, mu: float, sigma2:float):
        self.mu = mu
        self.sigma2 = sigma2
        if sigma2<=0:
            raise ValueError("σ^2 must be positive.")

    def mean(self):
        return self.mu

    def var(self):
        return self.sigma2

# Exponential Distribution
class ExpMoments(BaseDistributionMoments):
    pass

class GammaMoments(BaseDistributionMoments):
    pass

class BetaMoments(BaseDistributionMoments):
    pass

class CauchyMoments(BaseDistributionMoments):
    pass

class LogNormMoments(BaseDistributionMoments):
    pass

class Chi2Moments(BaseDistributionMoments):
    pass

class TMoments(BaseDistributionMoments):
    pass

class FMoments(BaseDistributionMoments):
    pass