import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import argparse

"""
Our core objective in designing Terra's stability mechanism is to contain volatility in Luna price changes.

Mining Rewards
Luna receives compensation for mining in the form of rewards. Rewards come in two forms:
- transaction fees: f(t)
- a portion of seigniorage: w(t)

Miining Rewards MR during period t are
MR(t) = f(t)*TV(t) + w(t)*max{ΔM(t),0}, where TV and M are Transaction Volume and Terra Money Supply respectively.

Pricing Luna
Luna has variable supply, so pricing one unit of Luna today is not as simple as discounting future rewards.
1 Luna may represent 1% of total rewards today and 0.5% tomorrow. We need to account for future fluctuations 
in supply in order to determine what share of each future rewards 1 Luna can claim today.

We can formulate the price of 1 Luna today using DCF as follows:
SUM for all t from now to infinity: Total Rewards(t)/Luna Supply(t)/(1+r)^t for an appropriate discount rate r.

The main benefit of this formulation is that it captures the idea that "the market prices in future dilution".
The market in fact ought to discount future rewards the more Luna Supply exceeds its target issuance.

Luna Price Volatility
Volatility in Luna price comes from volatility in future unit rewards, i.e. Total Rewards(t)/Luna Supply(t).
Unit rewards are by default highly cyclical: when the economy is growing rewards increase and supply tends
to decrease up to initial issuance; when the economy is shrinking rewards decrease and supply increases
as a result of the Terra contraction mechanism. Hence one way to contain Luna price volatility is to contain
volatility in unit rewards.

Countercyclical Mining Rewards
Our objective is to contain volatility in unit mining rewards, meaning Total Rewards(t)/Luna Supply(t). We have no
control over Luna Supply (simplifying here -- we do via buybacks but we can incorporate in mining rewards without
loss of generality). We *do* have control over Total Rewards.  The main intuition behind a countercyclical policy 
is that it attempts to counteract economic cycles by increasing rewards when the economy is in recession and 
decreasing rewards when the economy is growing.

We have two levers at our disposal to reduce volatility in mining rewards: transaction fees, and the proportion
of seigniorage that gets allocated to mining rewards. In what follows we implement this basic idea: adjust the
two levers to smooth out volatility in unit mining rewards.

Usage: +++++++++

Inputs

Parameters
TV growth
TV vol
absorption/smoothing factor?

Outputs
"""


# State
# t representing week (0 to 52*5-1): 10 years
# TV
# TMCAP
# seigniorage?
# LMCAP
# L Supply
# L price
# f fees
# w weight
# MR mining rewards
# a bunch of moving averages?

# Parameters and Constants
# TV growth
# TV vol
# velocity (26)
# Luna P/E ratio (random walk in range 1 to 200?)

V = 26 # annual Terra velocity
TOTAL_DAYS = 10*364
PERIOD = 7 # in days
PERIODS_PER_YEAR = int(364/PERIOD)
NUM_PERIODS = int(TOTAL_DAYS/PERIOD)

GENESIS_FEE = 0.1/100
GENESIS_SEIGNIORAGE_WEIGHT = 0.1
GENESIS_LUNA_SUPPLY = 100

LUNA_PE = 50 # TODO TEMPORARY -- make stochastic

def plot_results(df):
	# plot TV
	ax = df.loc[:, ['TV']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Transaction Volume ($)')

	# plot ΔΜ
	ax = df.loc[:, ['ΔM']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('ΔΜ ($)')

	# plot f and w
	ax = df.loc[:, ['f', 'w']].plot(secondary_y=['w'])
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Transaction Fee (%)')
	ax.right_ax.set_ylabel('Seigniorage Weight (%)')
	ax.set_ylim(0, 0.02)
	ax.right_ax.set_ylim(0, 1)
	y_ticks = ax.get_yticks()
	ax.set_yticklabels(['{:,.2%}'.format(y) for y in y_ticks])
	y_ticks_right = ax.right_ax.get_yticks()
	ax.right_ax.set_yticklabels(['{:,.2%}'.format(y) for y in y_ticks_right])

	# plot MR and MRL
	ax = df.loc[:, ['MR', 'MRL']].plot(secondary_y=['MRL'])
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Mining Rewards ($)')
	ax.right_ax.set_ylabel('Mining Rewards per Luna ($)')

	# plot LS
	ax = df.loc[:, ['LS']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Luna Supply')

	# plot M and LMC
	ax = df.loc[:, ['M', 'LMC']].plot(secondary_y=['LMC'])
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Terra Money Supply ($)')
	ax.right_ax.set_ylabel('Luna Market Cap ($)')

	# plot LRR
	ax = df.loc[:, ['LRR']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Luna Reserve Ratio')

	plt.show()

# Transaction Volume to Terra Money Supply
def tv_to_m(tv):
	annual_tv = tv*PERIODS_PER_YEAR
	return annual_tv/V

# Mining Rewards to Luna Market Cap
# TODO add randomness
def mr_to_lmc(mr):
	annual_mr = mr*PERIODS_PER_YEAR
	return annual_mr*LUNA_PE

# nothing special going on here
# separating to avoid lots of if-statements in state transition code
def set_genesis_state(df):
	tv = df.at[0,'TV']
	df.at[0,'M'] = tv_to_m(tv)
	df.at[0,'f'] = GENESIS_FEE
	df.at[0,'w'] = GENESIS_SEIGNIORAGE_WEIGHT
	df.at[0,'MR'] = df.at[0,'f']*df.at[0,'TV'] # seigniorage not defined at genesis
	df.at[0,'LMC'] = mr_to_lmc(df.at[0,'MR'])
	df.at[0,'LS'] = GENESIS_LUNA_SUPPLY
	df.at[0,'MRL'] = df.at[0,'MR']/df.at[0,'LS']
	# f and w are forward-computed for the following state
	df.at[1,'f'] = GENESIS_FEE
	df.at[1,'w'] = GENESIS_SEIGNIORAGE_WEIGHT


"""
Evaluate the state of the system at time period t. This is where all of the work happens.
Assumes t >= 1
Assumes states up to and including period t-1 have been already evaluated.
Assumes TV has already been set (independent variable).
Assumes f and w have already been set upon evaluation of state t-1.
This is because f and w are forward computed at all t for t+1.
"""
def evaluate_state(df, t):
	tv = df.at[t,'TV']
	df.at[t,'M'] = tv_to_m(tv)
	delta_m = df.at[t,'M'] - df.at[t-1,'M']
	df.at[t,'S'] = max(delta_m, 0)
	df.at[t,'MR'] = df.at[t,'f']*df.at[t,'TV'] + df.at[t,'w']*df.at[t,'S']
	df.at[t,'LMC'] = mr_to_lmc(df.at[t,'MR'])

	if delta_m >= 0: # expansion
		df.at[t,'LS'] = df.at[t-1,'LS']
	else: # contraction
		lp_prev = df.at[t-1,'LMC']/df.at[t-1,'LS'] # previous Luna price
		num_luna_issued = -delta_m/lp_prev
		df.at[t,'LS'] = df.at[t-1,'LS'] + num_luna_issued

	df.at[t,'MRL'] = df.at[t,'MR']/df.at[t,'LS']

	if t < NUM_PERIODS-1:
		df.at[t+1,'f'] = df.at[t,'f']
		df.at[t+1,'w'] = df.at[t,'w']

# TODO do not forward-project at the last period

# TODO how do we forward project TV? Do we need to?

def identity_update(f,w):
	return NotImplementedError()

def taylor_update(f, w):
	return NotImplementedError()

def mac_update(f, w):
	return NotImplementedError()


if __name__ == '__main__':
	t = range(0, NUM_PERIODS)
	# TODO model tv as GBM
	tv = np.logspace(0, 2, num=NUM_PERIODS, base=10) # exponential from 1 to 100

	df = pd.DataFrame(data = {'t': t, 'TV': tv})
	df['M'] = np.NaN # Terra Money Supply
	df['S'] = np.NaN # seigniorage
	df['f'] = np.NaN # TX fee
	df['w'] = np.NaN # seigniorage weight
	df['MR'] = np.NaN # Mining Rewards
	df['LMC'] = np.NaN # Luna Market Cap
	df['LS'] = np.NaN # Luna Supply
	df['MRL'] = np.NaN # Mining Rewards per Luna
	df.set_index('t', inplace=True)

	set_genesis_state(df) # t=0

	for t in range(1, NUM_PERIODS):
		evaluate_state(df, t)

	# compute some extra columns
	df['ΔM'] = df['M'] - df['M'].shift(1) # changes in M
	df['LRR'] = df['LMC']/df['M'] # Luna Reserve Ratio

	print(df)

	plot_results(df)

