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
lambda

Outputs
Spending Multiplier and Funding Weight timeseries for each firm
"""


LAMBDA = None # command-line argument


def spending_multiplier(tv_cur, tv_prev, w_prev):
	return max((tv_cur - tv_prev)/w_prev, 0) if w_prev > 0 else np.NaN


def sanitize(m):
	return 0 if np.isnan(m) else m


def spending_multiplier_sum(multipliers):
	return sum([sanitize(m) for m in multipliers])


def fill_row(df, i):
	row = df.iloc[i]
	tv1 = row['TV1']
	tv2 = row['TV2']
	tv3 = row['TV3']
	tv_sum = sum([tv1, tv2, tv3])
	sm1 = spending_multiplier(tv1, df.iloc[i-1]['TV1'], df.iloc[i-1]['w1']) if i > 0 else np.NaN
	sm2 = spending_multiplier(tv2, df.iloc[i-1]['TV2'], df.iloc[i-1]['w2']) if i > 0 else np.NaN
	sm3 = spending_multiplier(tv3, df.iloc[i-1]['TV3'], df.iloc[i-1]['w3']) if i > 0 else np.NaN
	sm_sum = spending_multiplier_sum([sm1, sm2, sm3])

	if sm_sum == 0:
		w1 = tv1/tv_sum
		w2 = tv2/tv_sum
		w3 = tv3/tv_sum
	else:
		w1 = LAMBDA*tv1/tv_sum + (1-LAMBDA)*sanitize(sm1)/sm_sum
		w2 = LAMBDA*tv2/tv_sum + (1-LAMBDA)*sanitize(sm2)/sm_sum
		w3 = LAMBDA*tv3/tv_sum + (1-LAMBDA)*sanitize(sm3)/sm_sum

	df.iloc[i] = {'TV1': tv1, 'TV2': tv2, 'TV3': tv3,
				  'SM1': sm1, 'SM2': sm2, 'SM3': sm3,
				  'w1': w1, 'w2': w2, 'w3': w3}


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('_lambda', type=float, help='funding weight parameter lambda')
	args = parser.parse_args()
	if args._lambda < 0 or args._lambda > 1:
		parser.error('lambda must be between 0 and 1 inclusive')
	LAMBDA = args._lambda

	tv1 = np.concatenate([np.linspace(1,100,15), np.linspace(100,90,5)])
	tv2 = np.concatenate([np.zeros(5), np.linspace(1,50,5), np.linspace(50, 10, 10)])
	tv3 = np.concatenate([np.zeros(15), np.linspace(10,70,5)])
	df = pd.DataFrame(data = {'TV1': tv1, 'TV2': tv2, 'TV3': tv3})
	df['SM1'] = np.NaN
	df['SM2'] = np.NaN
	df['SM3'] = np.NaN
	df['w1'] = np.NaN
	df['w2'] = np.NaN
	df['w3'] = np.NaN

	for i in range(len(df)):
		fill_row(df, i)

	print(df)

	ax = df.loc[:, ['TV1', 'TV2', 'TV3']].plot()
	ax.set_xlabel('time (months)')
	ax.set_ylabel('Transaction Volume ($ mm)')
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

