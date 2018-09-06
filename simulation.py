import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Model import Model
from util import *
import sys

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
	base_genesis_gmv = 5*10**3 # $5B
	base_growth_rates = [1.15]*3 + [1.07]*3+ [1.05]*6 + [1.03]*21 + [1.025]*30 + [1.02]*56

	base = generate_alliance_gmv_series(base_genesis_gmv*1, scale_growth_rates(base_growth_rates, 1.08))
	bear = generate_alliance_gmv_series(base_genesis_gmv*0.9, scale_growth_rates(base_growth_rates, 0.9))
	bull = generate_alliance_gmv_series(base_genesis_gmv*1.3, scale_growth_rates(base_growth_rates, 1.3))
	return {"base": base, "bear": bear, "bull": bull}

def fiat_reserve_ratio_series():
	reserve_ratios_quarterly = [1,0.95,0.9,0.8] + [0.6,0.5,0.4,0.3] + [0.2,0.17,0.14,0.12] + [0.1,0.09,0.085,0.08] + [0.075,0.071,0.068,0.066] + [0.065,0.061,0.058,0.056] + [0.055,0.053,0.052,0.051] + [0.05]*12
	reserve_ratios_monthly = stretch_linear(reserve_ratios_quarterly, 3) + 2*[0.05] # last 2 months
	return reserve_ratios_monthly

def build_model_args(scenario):
	gmv_scenarios = alliance_gmv_scenarios()
	fiat_reserve_ratios = fiat_reserve_ratio_series()
	target_discounts = [0.075]*6 + [0.05]*24 + [0.02]*90
	user_pen_ranges = {"base": (0.1,0.6), "bear": (0.05,0.4), "bull": (0.15,0.75)}
	return (gmv_scenarios[scenario], fiat_reserve_ratios, target_discounts, user_pen_ranges[scenario])

def plot_scenarios(m_base, m_bear, m_bull, attribute_args):
	for arg in attribute_args:
		attribute, annualize = arg

		t = np.linspace(0, 119, 120)
		base_timeseries = m_base.timeseries(attribute, annualize)
		bear_timeseries = m_bear.timeseries(attribute, annualize)
		bull_timeseries = m_bull.timeseries(attribute, annualize)

		plt.figure()
		plt.plot(t, bull_timeseries, "tab:green", label="bull")
		plt.plot(t, base_timeseries, "tab:blue", label="base")
		plt.plot(t, bear_timeseries, "tab:red", label="bear")
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

	attribute_args = [("user_penetration", False),
					  ("volume_penetration", False),
					  ("alliance_gmv", True),
					  ("nominal_volume", True),
					  ("discount", False),
					  ("hodl_ratio", False),
					  ("seigniorage", False),
					  ("terra_market_cap", False),
					  ("cash", False),
					  ("free_cash_flow", False),
					  ("target_reserve_ratio", False),
					  ("reserve_ratio", False)]
	plot_scenarios(m_base, m_bear, m_bull, attribute_args)

