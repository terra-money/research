import numpy as np
import matplotlib.pyplot as plt

"""
Generates a Geometric Brownian Motion timeseries.
Args:
    S0: the first term in the timeseries
    mu: the drift parameter of the GBM
    sigma: the volatility parameter of the GBM
    num_periods: the number of periods to model
    increments_per_period: the number of increments in the timeseries per period
"""
def gbm(S0, mu, sigma, num_periods, increments_per_period):
    T = num_periods
    dt = 1/increments_per_period
    N = num_periods*increments_per_period
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) # standard brownian motion
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) # geometric brownian motion
    return S

"""
Generates a cyclical timeseries, each cycle modeled independently by a Geometric Brownian Motion.
This is a useful function for simulating macroeconomic variables like GDP.
Args:
    S0: the first term in the timeseries
    mu_boom: the drift parameter of the GBM in a boom cycle (positive)
    mu_bust: the drift parameter of the GBM in a bust cycle (negative)
    sigma: the volatility parameter of the GBM
    cycle_lengths: an array of integers specifying the length of each cycle (number of periods)
    increments_per_period: the number of increments in the timeseries per period

cycle_lengths could be [2,4,3,4] for example, specifying four cycles with lenghts 2, 4, 3 and 4 periods respectively.
Cycles alternate between boom and bust. The first cycle is always a boom.
"""
def gbm_cyclical(S0, mu_boom, mu_bust, sigma, cycle_lengths, increments_per_period):
    cycle_lengths = np.array(cycle_lengths)
    cycle_lengths *= increments_per_period
    
    dt = 1
    mu_boom = mu_boom/increments_per_period
    mu_bust = mu_bust/increments_per_period
    sigma = np.sqrt(sigma**2/increments_per_period)

    cycles = np.array([[]])
    for i, cycle_length in enumerate(cycle_lengths):
        cycle_mu = mu_boom if i%2 == 0 else mu_bust
        cycle = np.exp((cycle_mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0, np.sqrt(dt), (1,cycle_length)))
        cycles = np.concatenate((cycles, cycle), axis=1)

    return S0*cycles.cumprod()

if __name__ == '__main__':
    # demonstrates the usage of gbm and gbm_cyclical
    np.random.seed(0)
    t = range(0, 520)
    S = gbm(1, 0.05, 0.5, 10, 52)
    S_cyclical = gbm_cyclical(1, 0.34, -0.2, 0.3, [2,3,5], 52)
    plt.plot(t, S, S_cyclical)
    plt.show()
