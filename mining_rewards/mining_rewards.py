import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import argparse

# common hack to import from sibling directory utils
# alternatives involve making all directories packages, creating setup files etc
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.gbm import gbm

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

The main benefit of this formulation is that it captures the idea that "the market prices in future dilution",
in the sense that given a projection of Luna Supply in the future they can soundly price 1 Luna today.

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
# seigniorage
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
NUM_YEARS = 10
TOTAL_DAYS = NUM_YEARS*364
PERIOD = 7 # in days
PERIODS_PER_YEAR = int(364/PERIOD)
NUM_PERIODS = int(TOTAL_DAYS/PERIOD)

PERIODS_PER_WINDOW = 13 # 13-week windows, ie 1 fiscal quarter

GENESIS_FEE = 0.1/100
GENESIS_SEIGNIORAGE_WEIGHT = 0.1
GENESIS_LUNA_SUPPLY = 100

GENESIS_LPE = 30

# GBM parameters for TV
MU = 0.34
SIGMA = 0.3

def plot_results(df):
	# plot TV
	ax = df.loc[:, ['TV', 'TV_MA']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Transaction Volume ($)')

	# plot ΔΜ
	ax = df.loc[:, ['ΔM', 'ΔM_MA']].plot()
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
	#ax = df.loc[:, ['MR_MA', 'MRL_MA']].plot(secondary_y=['MRL_MA'])
	ax = df.loc[:, ['MR_MA1', 'MR_MA2']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Mining Rewards ($)')
	#ax.right_ax.set_ylabel('Mining Rewards per Luna ($)')

	# plot LS
	ax = df.loc[:, ['LS']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Luna Supply')

	# plot LPE
	ax = df.loc[:, ['LPE']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Luna PE Ratio')

	# plot M and LMC
	ax = df.loc[:, ['M', 'LMC']].plot(secondary_y=['LMC'])
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Terra Money Supply ($)')
	ax.right_ax.set_ylabel('Luna Market Cap ($)')

	# plot LRR
	ax = df.loc[:, ['LRR', 'LRR_MA']].plot()
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Luna Reserve Ratio')

	# plot FMR
	ax = df.loc[:, ['FMR']].plot(kind='area')
	ax.set_xlabel('time (weeks)')
	ax.set_ylabel('Fee to Mining Reward Ratio')

	plt.show()

"""
Stochastic P/E multiple for Luna at time t, LPE(t)
Basic idea is to increase multiple during growth, decrease during recession
We model LPE(t) as (1 + X(t))*LPE(t-1), where X(t) is N(μ, σ):
μ is (MA1/MA2 - 1)/100, where MA1, MA2 are 1 and 2 year MAs for MR (earnings)
σ is 0.5% if MA1 >= MA2, otherwise it is 1%
Note that the updates are weekly, so eg 1% weekly vol is 7.2% annual vol

LPE is basically a random walk whose next value depends on the trend in MR
We make it more volatile when MR is in a downtrend
"""
# TODO may want to punish drops more by increasing negative mu's by 50-100%
def lpe(df, t):
	prev_lpe = df.at[t-1,'LPE']
	mr_ma1 = df['MR'].rolling(PERIODS_PER_YEAR, min_periods=1).mean().at[t]
	mr_ma2 = df['MR'].rolling(2*PERIODS_PER_YEAR, min_periods=1).mean().at[t]
	mr_delta = mr_ma1/mr_ma2 - 1
	mu = mr_delta/100
	sigma = 0.005 if mr_ma1 >= mr_ma2 else 0.01
	x = np.random.normal(mu, sigma)
	return (1 + x)*prev_lpe

# Transaction Volume to Terra Money Supply
def tv_to_m(tv):
	annual_tv = tv*PERIODS_PER_YEAR
	return annual_tv/V

# Mining Rewards to Luna Market Cap
# TODO add randomness
# TODO use MA rather than latest MR to smooth out
def mr_to_lmc(df, t):
	mr_ma = df['MR'].rolling(PERIODS_PER_WINDOW, min_periods=1).mean().at[t]
	annualized_mr = mr_ma*PERIODS_PER_YEAR
	lpe = df.at[t,'LPE']
	return annualized_mr*lpe

# nothing special going on here
# separating to avoid lots of if-statements in state transition code
def set_genesis_state(df):
	tv = df.at[0,'TV']
	df.at[0,'M'] = tv_to_m(tv)
	df.at[0,'f'] = GENESIS_FEE
	df.at[0,'w'] = GENESIS_SEIGNIORAGE_WEIGHT
	df.at[0,'MR'] = df.at[0,'f']*df.at[0,'TV'] # seigniorage not defined at genesis
	df.at[0,'LPE'] = GENESIS_LPE
	df.at[0,'LMC'] = mr_to_lmc(df, 0)
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
	df.at[t,'LPE'] = lpe(df, t)
	df.at[t,'LMC'] = mr_to_lmc(df, t)

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

# TODO how do we forward project TV? Do we need to? probably not given we are using MAs
# TODO enforce bounds on the magnitude of f and w changes per period (enforce on all
# functions?)

# where are MAs essential?

def identity_update(f,w):
	raise NotImplementedError()

def taylor_update(f, w):
	raise NotImplementedError()

def smooth_update(f, w):
	raise NotImplementedError()

def debt_update(f, w):
	raise NotImplementedError()

if __name__ == '__main__':
	np.random.seed(0) # for consistent outputs while developing
	t = range(0, NUM_PERIODS)
	tv = gbm(1, MU, SIGMA, NUM_YEARS, PERIODS_PER_YEAR)

	df = pd.DataFrame(data = {'t': t, 'TV': tv})
	df['M'] = np.NaN # Terra Money Supply
	df['S'] = np.NaN # seigniorage
	df['f'] = np.NaN # TX fee
	df['w'] = np.NaN # seigniorage weight
	df['MR'] = np.NaN # Mining Rewards
	df['LPE'] = np.NaN # Luna PE ratio
	df['LMC'] = np.NaN # Luna Market Cap
	df['LS'] = np.NaN # Luna Supply
	df['MRL'] = np.NaN # Mining Rewards per Luna
	df.set_index('t', inplace=True)

	set_genesis_state(df) # t=0

	for t in range(1, NUM_PERIODS):
		evaluate_state(df, t)

	# compute some extra columns

	# TODO plot fee to seigniorage revenue ratio -- do we want to smooth this out?
	df['ΔM'] = df['M'] - df['M'].shift(1) # changes in M
	df['LRR'] = df['LMC']/df['M'] # Luna Reserve Ratio

	rolling_fees = (df['f']*df['TV']).rolling(PERIODS_PER_WINDOW, min_periods=1).sum()
	rolling_mr = df['MR'].rolling(PERIODS_PER_WINDOW, min_periods=1).sum()
	df['FMR'] = rolling_fees/rolling_mr # cumulative fee to MR ratio, rolling quarterly

	df['TV_MA'] = df['TV'].rolling(PERIODS_PER_WINDOW, min_periods=1).mean()
	df['MR_MA1'] = df['MR'].rolling(52, min_periods=1).mean()
	df['MR_MA2'] = df['MR'].rolling(104, min_periods=1).mean()
	df['MRL_MA'] = df['MRL'].rolling(PERIODS_PER_WINDOW, min_periods=1).mean()
	df['ΔM_MA'] = df['ΔM'].rolling(PERIODS_PER_WINDOW, min_periods=1).mean()
	df['LRR_MA'] = df['LRR'].rolling(PERIODS_PER_WINDOW, min_periods=1).mean()

	print(df)

	plot_results(df)

