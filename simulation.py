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

"""
Returns:
	numpy array -- time series of quarterly Terra Alliance GMV
"""
def alliance_gmv_series_plot(base, bear, bull):
	t = np.linspace(0, 119, 120)

	plt.figure()
	plt.plot(t, base, "tab:blue", label="base")
	plt.plot(t, bear, "tab:red", label="bear")
	plt.plot(t, bull, "tab:green", label="bull")	
	plt.title("Alliance GMV (annualized)")
	plt.show()

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

def plot_scenarios(m_base, m_bear, m_bull, attribute_args):
	for arg in attribute_args:
		attribute, y_type, annualize = arg

		t = np.linspace(0, 119, 120)
		base_timeseries = m_base.timeseries(attribute, annualize)
		bear_timeseries = m_bear.timeseries(attribute, annualize)
		bull_timeseries = m_bull.timeseries(attribute, annualize)

		t_dates = [START_DATE + relativedelta(months=m) for m in t]
		years = mdates.YearLocator()   # every year
		yearsFmt = mdates.DateFormatter('%Y')
		fig, ax = plt.subplots()

		# type-specific formatting:
		if y_type == "%":
			bull_timeseries *= 100
			base_timeseries *= 100
			bear_timeseries *= 100
			ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
		elif y_type == "$":
			# convert from $mm to $B
			bull_timeseries /= 1000
			base_timeseries /= 1000
			bear_timeseries /= 1000
			plt.ylabel("$B")
		else:
			raise RuntimeError("unrecognized y_type")

		ax.plot(t_dates, bull_timeseries, "tab:green", label="bull")
		ax.plot(t_dates, base_timeseries, "tab:blue", label="base")
		ax.plot(t_dates, bear_timeseries, "tab:red", label="bear")

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

		plt.title(attribute + (" (annualized)" if annualize else ""))
		leg = plt.legend(loc='best')
		leg.get_frame().set_alpha(0.5)
		plt.show()

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

	attribute_args = [("base_volume_penetration", "%", False),
					  ("volume_penetration", "%", False),
					  ("alliance_gmv", "$", True),
					  ("nominal_volume", "$", True),
					  ("discount", "%", False),
					  ("free_cash_flow", "$", False),
					  ("seigniorage", "$", False),
					  ("reserve_ratio", "%", False)]

	print("\nAnnual Dividends")
	print(m_bull.timeseries_annual("dividends"))
	print(m_base.timeseries_annual("dividends"))
	print(m_bear.timeseries_annual("dividends"))
	print("\nCumulative FCFF")
	print(m_bull.timeseries_annual("free_cash_flow"))
	print(m_base.timeseries_annual("free_cash_flow"))
	print(m_bear.timeseries_annual("free_cash_flow"))
	#plot_scenarios(m_base, m_bear, m_bull, attribute_args)

