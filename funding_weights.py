import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
Simulates governance funding weights in a 3 firm economy.

Usage
python funding_weights.py lambda, where lambda is a float between 0 and 1 inclusive
e.g. for lambda = 0.5 you would run:
python funding_weights.py 0.5

Inputs
Transaction Volume timeseries for each firm

Parameters
lambda

Outputs
Spending Multiplier timeseries for each firm 
Funding Weight timeseries for each firm

See "Funding Weight Simulation" on Quip for the funding equation, simulation results and discussion.
"""


LAMBDA = None # command-line argument set in the main function


def spending_multiplier(tv_cur, tv_prev, w_prev):
	return max((tv_cur - tv_prev)/w_prev, 0) if w_prev > 0 else np.NaN

"""
Convert np.nan to 0 for ensuing computation
"""
def sanitize(m):
	return 0 if np.isnan(m) else m

"""
Sum spending multipliers after sanitizing them
"""
def spending_multiplier_sum(multipliers):
	return sum([sanitize(m) for m in multipliers])

"""
Evaluate the state of the system at time period t
Assumes states up to period t-1 have been already evaluated
"""
def evaluate_state(df, t):
	row = df.iloc[t]
	tv1 = row['TV1']
	tv2 = row['TV2']
	tv3 = row['TV3']
	tv_sum = sum([tv1, tv2, tv3])

	# compute spending multipliers by referring to previous TV and weight
	# the spending multiplier is not defined for the first period
	sm1 = spending_multiplier(tv1, df.iloc[t-1]['TV1'], df.iloc[t-1]['w1']) if t > 0 else np.NaN
	sm2 = spending_multiplier(tv2, df.iloc[t-1]['TV2'], df.iloc[t-1]['w2']) if t > 0 else np.NaN
	sm3 = spending_multiplier(tv3, df.iloc[t-1]['TV3'], df.iloc[t-1]['w3']) if t > 0 else np.NaN
	sm_sum = spending_multiplier_sum([sm1, sm2, sm3])

	# full funding equation is used only when there is at least one SM that is defined and non-zero
	if sm_sum == 0:
		w1 = tv1/tv_sum
		w2 = tv2/tv_sum
		w3 = tv3/tv_sum
	else:
		w1 = LAMBDA*tv1/tv_sum + (1-LAMBDA)*sanitize(sm1)/sm_sum
		w2 = LAMBDA*tv2/tv_sum + (1-LAMBDA)*sanitize(sm2)/sm_sum
		w3 = LAMBDA*tv3/tv_sum + (1-LAMBDA)*sanitize(sm3)/sm_sum

	df.iloc[t] = {'TV1': tv1, 'TV2': tv2, 'TV3': tv3,
				  'SM1': sm1, 'SM2': sm2, 'SM3': sm3,
				  'w1': w1, 'w2': w2, 'w3': w3}


if __name__ == "__main__":
	# read λ from the command line
	parser = argparse.ArgumentParser()
	parser.add_argument('_lambda', type=float, help='funding weight parameter lambda')
	args = parser.parse_args()
	if args._lambda < 0 or args._lambda > 1:
		parser.error('lambda must be between 0 and 1 inclusive')
	LAMBDA = args._lambda

	# sample input TV timeseries
	tv1 = np.concatenate([np.linspace(1,100,15), np.linspace(100,90,5)])
	tv2 = np.concatenate([np.zeros(5), np.linspace(1,50,5), np.linspace(50, 10, 10)])
	tv3 = np.concatenate([np.zeros(15), np.linspace(10,70,5)])

	# organize data in pandas DataFrame with columns for time period, TVs, SMs and weights
	df = pd.DataFrame(data = {'TV1': tv1, 'TV2': tv2, 'TV3': tv3})
	df['SM1'] = np.NaN
	df['SM2'] = np.NaN
	df['SM3'] = np.NaN
	df['w1'] = np.NaN
	df['w2'] = np.NaN
	df['w3'] = np.NaN

	# evaluate state for each time period
	for t in range(len(df)):
		evaluate_state(df, t)

	print(df)
	df.to_csv('funding_weights_output_{}.csv'.format(LAMBDA), index=False)

	ax = df.loc[:, ['TV1', 'TV2', 'TV3']].plot()
	ax.set_xlabel('time (months)')
	ax.set_ylabel('Transaction Volume ($ mm)')
	ax.set_xlim(0, 20)

	ax = df.loc[:, ['SM1', 'SM2', 'SM3']].plot(logy=True,title='λ={}'.format(LAMBDA)) # note that the SMs depend on λ
	ax.set_xlabel('time (months)')
	ax.set_ylabel('Spending Multiplier (log)')
	ax.set_xlim(0, 20)

	ax = df.loc[:, ['w1', 'w2', 'w3']].plot(kind='area',title='λ={}'.format(LAMBDA))
	ax.set_xlabel('time (months)')
	ax.set_ylabel('Funding Weight')
	ax.set_xlim(0, 20)

	ax = df.loc[:, ['w1', 'w2', 'w3']].plot(kind='area', stacked=False, title='λ={}'.format(LAMBDA))
	ax.set_xlabel('time (months)')
	ax.set_ylabel('Funding Weight')
	ax.set_xlim(0, 20)

	plt.show()

