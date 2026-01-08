import sys
import pytest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import warnings
warnings.filterwarnings('ignore')

import numpy as np


from src.distribution_moments import (
    DiscreteUniformMoments,
    BernoulliMoments,
    BinomialMoments,
    HyperGeoMoments,
    PoissonMoments,
    GeometricMoments,
    NBMoments,
    ContinuousUniformMoments,
    NormalMoments,
    LogNormalMoments,
    ExpMoments,
    GammaMoments,
    BetaMoments,
    Chi2Moments,
    TMoments,
    FMoments
)

# Discrete Uniform Distribution
def test_does_discrete_uniformmoments_happen_valueerror():
    try:
        dum = DiscreteUniformMoments(1, 0)
    except ValueError as e:
        print(e)
        print("OK")

def test_is_mean_of_discrete_uniform_distribution_correct():
    dum = DiscreteUniformMoments(0, 1)
    print(dum.mean())

def test_is_variance_of_discrete_uniform_distribution_correct():
    dum = DiscreteUniformMoments(0, 1)
    print(dum.var())

# Bernoulli distribution
def test_is_parameter_of_Bernoulli_distribution_correct():
    import numpy as np
    ps = [0.2, 1.2, 0.54, -3.2, np.pi/4, -0.4, 0.8]
    for p in ps:
        try:
            ber = BernoulliMoments(p)
        except ValueError as e:
            print(e)

def test_is_mean_of_Bernoulli_distribution_correct():
    p = 0.7
    ber = BernoulliMoments(p)
    return ber.mean()

def test_is_variance_of_Bernoulli_distribution_correct():
    p = 0.7
    ber = BernoulliMoments(p)
    return ber.var()


# binomial distribution
def test_are_parameters_of_binomial_distribution_correct():
    import numpy as np
    ns = [1, 0.2, 4, 3, 5, -2, -2.5]
    ps = [0.2, 1.2, 0.54, -3.2, np.pi/4, -0.4, 0.8]
    for n, p in zip(ns, ps):
        try:
            bi = BinomialMoments(n, p)
        except ValueError as e:
            print(e)

def test_is_mean_of_binomial_distribution_correct():
    n = 4
    p = 0.7
    bi = BinomialMoments(n, p)
    return bi.mean()


def test_is_variance_of_binomial_distribution_correct():
    n = 4
    p = 0.7
    bi = BinomialMoments(n, p)
    return bi.var()


# HyperGeometric distribution
def test_are_parameters_of_hypergeo_distribution_correct():
    import numpy as np
    Ns = [1, -1, 4, 3, 5, -2, -2.5]
    Ms = [2, 2, 3, -2, np.pi, -0.4, 0.8]
    ns = [1, 2, 1, -1, np.exp(1), 6, 7]
    for N, M, n in zip(Ns, Ms, ns):
        try:
            hg = HyperGeoMoments(N, M, n)
        except ValueError as e:
            print(e)

def test_is_mean_of_hypergeo_distribution_correct():
    N = 20
    M = 7
    n = 2
    hg = HyperGeoMoments(N, M, n)
    return hg.mean()

def test_is_variance_of_hypergeo_distribution_correct():
    N = 20
    M = 7
    n = 2
    hg = HyperGeoMoments(N, M, n)
    return hg.var()

# Poisson distribution
def test_are_parameters_of_poisson_distribution_correct():
    import numpy as np
    Ls = [0, 3, -1, 2.5]
    for L in Ls:
        try:
            po = PoissonMoments(L)
        except ValueError as e:
            print(e)

def test_is_mean_of_poisson_distribution_correct():
    L = 2.5
    po = PoissonMoments(L)
    return po.mean()

def test_is_variance_of_poisson_distribution_correct():
    L = 2.5
    po = PoissonMoments(L)
    return po.var()

# Geometric distribution
def test_are_parameters_of_geo_distribution_correct():
    ps = [0.1, 1.5, -0.4, np.pi]
    for p in ps:
        try:
            geo = GeometricMoments(p)
        except ValueError as e:
            print(e)

def test_is_mean_of_geo_distribution_correct():
    p = 0.3
    geo = GeometricMoments(p)
    return geo.mean()

def test_is_variance_of_geo_distribution_correct():
    p = 0.3
    geo = GeometricMoments(p)
    return geo.var()


# Negative binomial distribution
def test_are_parameters_of_negative_binomial_distribution_correct():
    import numpy as np
    rs = [1, 0.2, 4, 3, 5, -2, -2.5]
    ps = [0.2, 1.2, 0.54, -3.2, np.pi/4, -0.4, 0.8]
    for r, p in zip(rs, ps):
        try:
            nb = NBMoments(r, p)
        except ValueError as e:
            print(e)

def test_is_mean_of_negative_binomial_distribution_correct():
    r = 4
    p = 0.7
    nb = NBMoments(r, p)
    return nb.mean()


def test_is_variance_of_negative_binomial_distribution_correct():
    r = 4
    p = 0.7
    nb = NBMoments(r, p)
    return nb.var()

# Continuous Uniform Distribution
def test_does_continuous_uniform_distribution_happen_valueerror():
    try:
        cum = ContinuousUniformMoments(1, 0)
    except ValueError as e:
        print(e)
        print("OK")

def test_is_mean_of_continuous_uniform_distribution_correct():
    cum = ContinuousUniformMoments(0, 1)
    return cum.mean()

def test_is_variance_of_continuous_uniform_distribution_correct():
    cum = ContinuousUniformMoments(0, 1)
    return cum.var()

# Normal Distribution
def test_does_normal_distribution_happen_valueerror():
    mu = 0
    sigma2s = [0.2, 0, -1, 15]
    for sigma2 in sigma2s:
        try:
            norm = NormalMoments(mu, sigma2)
        except ValueError as e:
            print(e)
            print("OK")

def test_is_mean_of_normal_distribution_correct():
    mu = 0
    sigma2 = 1
    norm = NormalMoments(mu, sigma2)
    return norm.mean()

def test_is_variance_of_normal_distribution_correct():
    mu = 0
    sigma2 = 1
    norm = NormalMoments(mu, sigma2)
    return norm.var()


# Log-Normal Distribution
def test_does_log_normal_distribution_happen_valueerror():
    mu = 0
    sigma2s = [0.2, 0, -1, 15]
    for sigma2 in sigma2s:
        try:
            lognorm = LogNormalMoments(mu, sigma2)
        except ValueError as e:
            print(e)
            print("OK")
    

def test_is_mean_of_log_normal_distribution_correct():
    mu = 0
    sigma2 = 1
    lognorm = LogNormalMoments(mu, sigma2)
    return lognorm.mean()

def test_is_variance_of_log_normal_distribution_correct():
    mu = 0
    sigma2 = 1
    lognorm = LogNormalMoments(mu, sigma2)
    return lognorm.var()


def test_does_exponential_distribution_happen_valueerror():
    Ls = [0.2, 0, -1, 15]
    for L in Ls:
        try:
            exp = ExpMoments(L)
        except ValueError as e:
            print(e)
            print("OK")

def test_is_mean_of_exponential_distribution_correct():
    L = 2
    norm = ExpMoments(L)
    return norm.mean()

def test_is_variance_of_exponential_distribution_correct():
    L = 2
    norm = ExpMoments(L)
    return norm.mean()


def test_does_gamma_distribution_happen_valueerror():
    alphas = [1, 0.2, 4, 3, 5, -2, -2.5]
    betas = [0.2, 1.2, 0.54, -3.2, np.pi/4, -0.4, 0.8]
    for alpha, beta in zip(alphas, betas):
        try:
            gamma = GammaMoments(alpha, beta)
        except ValueError as e:
            print(e)
            print("OK")

def test_is_mean_of_gamma_distribution_correct():
    alpha = 2
    beta = 3
    gamma = GammaMoments(alpha, beta)
    return gamma.mean()

def test_is_variance_of_gamma_distribution_correct():
    alpha = 2
    beta = 3
    gamma = GammaMoments(alpha, beta)
    return gamma.var()


def test_does_beta_happen_valueerror():
    alphas = [1, 0.2, 4, 3, 5, -2, -2.5]
    bs = [0.2, 1.2, 0.54, -3.2, np.pi/4, -0.4, 0.8]
    for alpha, b in zip(alphas, bs):
        try:
            beta = BetaMoments(alpha, b)
        except ValueError as e:
            print(e)
            print("OK")

def test_is_mean_of_beta_distribution_correct():
    alpha = 2
    b = 3
    beta = BetaMoments(alpha, b)
    return beta.mean()

def test_is_variance_of_beta_distribution_correct():
    alpha = 2
    b = 3
    beta = BetaMoments(alpha, b)
    return beta.mean()


def test_does_chi2_happen_valueerror():
    ks = [1, 0.2, 4, 3, 5, -2, -2.5]
    
    for k in ks:
        try:
            chi2 = Chi2Moments(k)
        except ValueError as e:
            print(e)
            print("OK")

def test_is_mean_of_chi2_distribution_correct():
    k = 2
    chi2 = Chi2Moments(k)
    return chi2.mean()

def test_is_variance_of_chi2_distribution_correct():
    k = 2
    chi2 = Chi2Moments(k)
    return chi2.mean()


def test_does_t_happen_valueerror():
    ms = [1, 0.2, 4, 3, 5, -2, -2.5]
    
    for m in ms:
        try:
            t = TMoments(m)
        except ValueError as e:
            print(e)
            print("OK")

def test_is_mean_of_t_distribution_correct():
    ms = [m for m in range(4)]
    for m in ms:
        try:
            t = TMoments(m)
            print(t.mean())
        except ValueError as e:
            print(e)
            print("OK")

def test_is_variance_of_t_distribution_correct():
    ms = [m for m in range(4)]
    for m in ms:
        try:
            t = TMoments(m)
            print(t.var())
        except ValueError as e:
            print(e)
            print("OK")


def test_does_F_happen_valueerror():
    ns = [1, 0.2, 4, 3, 5, -2, -2.5]
    m = 3
    for n in ns:
        try:
            f = FMoments(m, n)
        except ValueError as e:
            print(e)
            print("OK")

def test_is_mean_of_F_distribution_correct():
    ns = [n for n in range(4)]
    m = 3
    for n in ns:
        try:
            f = FMoments(m, n)
            print(f.mean())
        except ValueError as e:
            print(e)
            print("OK")

def test_is_variance_of_t_distribution_correct():
    ns = [n for n in range(6)]
    m = 3
    for n in ns:
        try:
            f = FMoments(m, n)
            print(f.var())
        except ValueError as e:
            print(e)
            print("OK")