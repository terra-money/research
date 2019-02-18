import numpy as np
import matplotlib.pyplot as plt

"""
Generates a Geometric Brownian Motion timeseries
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

if __name__ == '__main__':
	# demonstrates the usage of gbm
	np.random.seed(0)
	t = range(0, 520)
	S = gbm(1, 0.34, 0.3, 10, 52)
	plt.plot(t, S)
	plt.show()