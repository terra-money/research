import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
from Model import Model
from util import *
import sys
import datetime
from dateutil.relativedelta import relativedelta

START_DATE = datetime.date(2019,1,1)

def generate_alliance_gmv_series(genesis_gmv, growth_rates):
	gmv_series = [genesis_gmv]
	for r in growth_rates:
		gmv_series.append(gmv_series[-1]*r)
	return gmv_series

"""
Returns:
	tuple (base, bear, bull) timeseries, each of length NUM_MONTHS
"""
def alliance_gmv_scenarios():
	base_genesis_gmv = 25*10**3 # $5B
	base_growth_rates = [1.15]*3 + [1.07]*3+ [1.05]*6 + [1.03]*21 + [1.025]*30 + [1.02]*56

	# base = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 1.05))
	# bear = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 0.9))
	# bull = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 1.2))
	# bear = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 0.41)) # 15% CAGR
	bear = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 0.535)) # 20% CAGR
	base = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 0.66)) # 25% CAGR
	bull = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 0.775)) # 30% CAGR
	# bull = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 0.85)) # 35% CAGR
	return {"base": base, "bull": bull, "bear": bear}

def base_volume_penetration_scenarios():
	t = np.linspace(0,119,120)
	base = 0.015 + sigmoid(t, 80, 0.055, 0.30)
	bull = 0.03 + sigmoid(t, 80, 0.05, 0.45)
	bear = sigmoid(t, 80, 0.06, 0.15)
	# base = 0.015 + sigmoid(t, 80, 0.055, 0.30)
	# bull = 0.015 + sigmoid(t, 80, 0.055, 0.30)
	# bear = 0.015 + sigmoid(t, 80, 0.055, 0.30)
	return {"base": base, "bull": bull, "bear": bear}

def fiat_reserve_ratio_series():
	reserve_ratios_quarterly = [1,0.95,0.85,0.7] + [0.6,0.5,0.4,0.3] + [0.2,0.17,0.14,0.12] + [0.1,0.09,0.085,0.08] + [0.075,0.071,0.068,0.066] + [0.065,0.061,0.058,0.056] + [0.055,0.053,0.052,0.051] + [0.05]*12
	reserve_ratios_monthly = stretch_linear(reserve_ratios_quarterly, 3) + 2*[0.05] # last 2 months
	return reserve_ratios_monthly

def target_discount_series():
	target_discounts = np.linspace(0.1,0.05,7).tolist()[:-1] + np.linspace(0.05,0.04,25).tolist()[:-1] + np.linspace(0.04,0.03,19).tolist()[:-1] + np.linspace(0.03,0.02,23).tolist()[:-1] + [0.02]*50
	#target_discounts = [0.1]*3 + [0.075]*3 + [0.05]*24 + [0.04]*12 + [0.03]*8 + [0.02]*70
	#target_discounts = [0.075]*6 + [0.05]*24 + [0.02]*90
	return target_discounts

def build_model_args(scenario):
	gmv_scenarios = alliance_gmv_scenarios()
	bvp_scenarios = base_volume_penetration_scenarios()
	fiat_reserve_ratios = fiat_reserve_ratio_series()
	target_discounts = target_discount_series()
	return (gmv_scenarios[scenario], bvp_scenarios[scenario], fiat_reserve_ratios, target_discounts)

"""
Args:
	timeseries_list: a list of timeseries
	y_type: one of "%" or "$", describes all timeseries

	Timeseries of type "%" are provided as fractions, eg 0.1 for 10%
	Timeseries of type "$" are provided in $mm
"""
def preprocess_timeseries(timeseries_list, y_type):
	for ts in timeseries_list:
		if y_type == "%":
			ts *= 100
		elif y_type == "$":
			ts /= 1000 # convert from $mm to $B
		else:
			raise RuntimeError("unrecognized y_type: {}".format(y_type))

"""
Generic facility for custom plots of multiple timeseries

Args:
	timeseries_args: List of timeseries argument tuples, to be described below
	y_type: one of "%" or "$", describes all timeseries
	title: string to be used as title of the plot

	A timeseries argument is a tuple of the form (ts, label, color):
		ts is a timeseries of type y_type and length 120
		label is the string to be used in the legend for this timeseries, eg "Dividends"
		color is the string used for coloring the plot, eg "red"

		Timeseries of type "%" are provided as fractions, eg 0.1 for 10%
		Timeseries of type "$" are provided in $mm
"""
def plot_many(timeseries_args, y_type, title):
	# boilerplate plot setup...
	t = np.linspace(0, 119, 120)
	t_dates = [START_DATE + relativedelta(months=m) for m in t]
	years = mdates.YearLocator()   # every year
	yearsFmt = mdates.DateFormatter('%Y')
	fig, ax = plt.subplots()
	# format the ticks
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(yearsFmt)
	# round to nearest years...
	datemin = np.datetime64(t_dates[0], 'Y')
	datemax = np.datetime64(t_dates[-1], 'Y') + np.timedelta64(1, 'Y')
	ax.set_xlim(datemin, datemax)
	# rotates and right aligns the x labels, and moves the bottom of the
	# axes up to make room for them
	fig.autofmt_xdate()

	# type-specific formatting:
	if y_type == "%":
		ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
	elif y_type == "$":
		plt.ylabel("$B")

	preprocess_timeseries(list(zip(*timeseries_args))[0], y_type)

	for ts_arg in timeseries_args:
		ts = ts_arg[0]
		label = ts_arg[1]
		color = ts_arg[2]
		ax.plot(t_dates, ts, "tab:{}".format(color), label=label)

	plt.title(title)
	leg = plt.legend(loc='best')
	leg.get_frame().set_alpha(0.5)
	plt.show()

"""

"""
def plot_scenarios(models, attribute, y_type, annualize):
	bull_ts = models["bull"].timeseries(attribute, annualize)
	base_ts = models["base"].timeseries(attribute, annualize)
	bear_ts = models["bear"].timeseries(attribute, annualize)

	timeseries_args = [(bull_ts, "bull", "green"), (base_ts, "base", "blue"), (bear_ts, "bear", "red")]
	title = attribute + (" (annualized)" if annualize else "")
	plot_many(timeseries_args, y_type, title)



if __name__ == '__main__':
	bull_model_args = build_model_args("bull")
	m_bull = Model(*bull_model_args)
	m_bull.run()

	base_model_args = build_model_args("base")
	m_base = Model(*base_model_args)
	m_base.run()

	bear_model_args = build_model_args("bear")
	m_bear = Model(*bear_model_args)
	m_bear.run()

	print("\n\nBull Temperature: {:6,.0f} mm celsius".format(m_bull.valuation()))
	print("Base Temperature: {:6,.0f} mm celsius".format(m_base.valuation()))
	print("Bear Temperature: {:6,.0f} mm celsius".format(m_bear.valuation()))

	models = {"bull": m_bull, "base": m_base, "bear": m_bear}
	plot_scenarios(models, "base_volume_penetration", "%", False)

	# attribute_args = [("base_volume_penetration", "%", False),
	# 				  ("volume_penetration", "%", False),
	# 				  ("alliance_gmv", "$", True),
	# 				  ("nominal_volume", "$", True),
	# 				  ("discount", "%", False),
	# 				  ("free_cash_flow", "$", False),
	# 				  ("seigniorage", "$", False),
	# 				  ("reserve_ratio", "%", False)]



