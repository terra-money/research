import math
import numpy as np

# TODO comments needed :)

"""
Args:
	x: numpy array
	mid: x-value of sigmoid's midpoint
	k: steepness
	L: max value of curve
Returns:
	sigmoid numpy array
"""
def sigmoid(x, mid, k, L):
	y = k*(x-mid)
	return L / (1 + np.exp(-y))

"""
"""
def stretch_linear(array, k):
	stretched = []
	for i in range(len(array)-1): # exclude last element
		step = (array[i+1] - array[i])/k
		for j in range(k):
			stretched.append(array[i] + j*step)
	stretched.append(array[-1])
	return stretched

"""
"""
def stretch_geom(array, k):
	stretched = []
	for i in range(len(array)-1): # exclude last element
		step = math.pow(array[i+1]/array[i], 1/k)
		for j in range(k):
			stretched.append(array[i]*step**j)
	stretched.append(array[-1])
	return stretched

"""
"""
def arange_geom(a, b, k, inverted=False):
	x = b-a+1
	k -= 1 # for simplicity of proceeding calculations
	r = math.pow(x, 1/k)
	v = [1]
	for i in range(1, k+1):
		exp = k-i if inverted else i-1
		delta = (r**exp)*(r-1)
		v.append(v[i-1] + delta)
	return np.array(v) + a - 1


"""
"""
def scale_growth_rates(growth_rates, scale_factor):
	return [1 + (r-1)*scale_factor for r in growth_rates]

"""
We need a model that takes in a CAGR and a decay parameter
and returns a list of growth rates.
"""



