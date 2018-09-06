import numpy as np
from State import State
from util import *
import matplotlib
import matplotlib.pyplot as plt
import time
import math

# all monetary amounts quoted in $mm
MONEY_VELOCITY_MONTH = 1
TX_FEE = 0.005 # fraction
GENESIS_CASH = 100
FIAT_TO_TERRA_FEE = 0.01 # fraction
NUM_MONTHS = 120
GMV_USER_YEAR = 4*10**(-4) # average GMV per user per year -- $400 in $mm

"""
The goal is to make all independent variables explicit:
- Alliance GMV
- user penetration range
- target reserve ratios
- target discounts
A Model maintains a list of past States as well as the current State

Notes and observations:
- Seigniorage = Real Volume Delta + HODL rate
- Free Cash FLow
- Dividends
- Cash

Dividends = Tx Fees
FCFF = Seigniorage - Fees - Target Fiat Reserve Allocation

- All monetary amounts quoted in $mm
- Volumes, fees etc are all monthly
"""

class Model:

	"""
	Initialize Model with independent variables.
	alliance_gmvs are annualized, we convert to monthly
	"""
	def __init__(self,
				alliance_gmvs,
				target_reserve_ratios,
				target_discounts,
				user_penetration_range):
		self.state = State(cash = GENESIS_CASH)
		self.state_history = [] # don't include genesis state
		self.alliance_gmvs = np.array(alliance_gmvs)/12 # convert to monthly
		self.target_reserve_ratios = target_reserve_ratios
		self.target_discounts = target_discounts
		self.user_penetration_range = user_penetration_range

	def _alliance_user_model(self, alliance_gmv):
		return alliance_gmv*12/GMV_USER_YEAR

	def _user_penetration_model(self, month, user_penetration_range):
		p_min, p_max = user_penetration_range
		return p_min + sigmoid(month, 30, 0.4/3, p_max - p_min)

	def _base_user_volume_capture(self, month):
		l = np.linspace(0, 75, 120)/100
		g = arange_geom(0, 75, 120)/100
		v = (l+g)/2
		return v[month]

	def _discount_to_volume_capture(self, discount):
		components = [arange_geom(0,50,21),arange_geom(50,52.5,11,inverted=True)[1:],arange_geom(52.5,75,21)[1:],
					  arange_geom(75,80,21,inverted=True)[1:],arange_geom(80,90,31)[1:]]
		y = np.concatenate(components)/100
		idx = int(discount*1000)
		return y[idx]

	"""
	Fraction of our users' volume we are able to capture.
	Depends on two variables: month and discount
	Month informs base volume capture, bvc
	Discount informs discount volume capture for *non-base* volume, dvc
	"""
	def _user_volume_capture_model(self, month, discount):
		bvc = self._base_user_volume_capture(month)
		dvc = self._discount_to_volume_capture(discount)
		return bvc + (1-bvc)*dvc


	def _volume_penetration_model(self, month, user_penetration, discount):
		return user_penetration*self._user_volume_capture_model(month, discount)

	"""
	Store of value ratio -- linearly grows between 0.1 and 0.4 over 10 years
	For context note that USD's M2 is almost 4x M1 (M2 including M1 as well
	as savings deposits and MMFs)
	"""
	def _hodl_ratio_model(self, month):
		l = np.linspace(0, 40, 120)/100
		g = arange_geom(0, 40, 120)/100
		h = (l+g)/2
		return h[month]

	"""
	Transitions system to next state -- note that state represents the end of the period
	"""
	def _transition(self,
					next_alliance_gmv,
					next_target_reserve_ratio,
					next_target_discount,
					user_penetration_range):
		current = self.state # current State
		next = State()
		next.month = current.month + 1
		next.cash = current.cash

		# users and volume
		next.alliance_gmv = next_alliance_gmv
		next.alliance_users = self._alliance_user_model(next.alliance_gmv)
		next.user_penetration = self._user_penetration_model(next.month, user_penetration_range)
		next.volume_penetration = self._volume_penetration_model(next.month, next.user_penetration, next_target_discount)
		next.nominal_volume = next.volume_penetration*next.alliance_gmv
 
		# fees
		next.target_discount = next_target_discount
		next.discount = next.target_discount
		discount_fees = next_target_discount*next.nominal_volume
		next.real_volume = next.nominal_volume - discount_fees
		fiat_conversion_fees = next.real_volume*FIAT_TO_TERRA_FEE
		fees = discount_fees + fiat_conversion_fees
		next.cash -= fees
		assert next.cash > 0, next

		# seigniorage
		commerce_seigniorage = (next.real_volume - current.real_volume)/MONEY_VELOCITY_MONTH # may be negative
		next.terra_commerce_market = current.terra_commerce_market + commerce_seigniorage
		next.hodl_ratio = self._hodl_ratio_model(next.month)
		next.terra_market_cap = next.terra_commerce_market/(1 - next.hodl_ratio)
		next.seigniorage = next.terra_market_cap - current.terra_market_cap # may be negative
		next.cash += next.seigniorage
		assert next.cash > 0, next

		# reserve
		next.target_reserve_ratio = next_target_reserve_ratio
		next.reserve_ratio = next.cash/next.terra_market_cap
		assert next.reserve_ratio >= next.target_reserve_ratio, next

		# dividends and Free Cash Flow
		next.dividends = next.real_volume*TX_FEE # paid in Terra
		# to obtain Free Cash Flow, follow 3 steps:
		# - add cash generated that is not locked up in Fiat Reserve
		# - add cash in Fiat Reserve that becomes available whenever the target reserve ratio drops
		# - subtract fees
		next.free_cash_flow = next.seigniorage*(1 - next.target_reserve_ratio) + current.terra_market_cap*(current.target_reserve_ratio - next.target_reserve_ratio) - fees

		self.state = next
		self.state_history.append(self.state)


	def run(self):
		args = zip(self.alliance_gmvs, self.target_reserve_ratios, self.target_discounts, [self.user_penetration_range]*NUM_MONTHS)
		for arg in args:
			time.sleep(0.01)
			self._transition(*arg)
			print(self.state)


	def valuation(self):
		fcff = np.array([s.free_cash_flow + s.dividends for s in self.state_history])
		d = np.logspace(1, NUM_MONTHS, num=NUM_MONTHS, base=self._discount_rate())
		comp_period_value =  np.sum(fcff/d)
		term_value = self._terminal_value()
		return comp_period_value + term_value

	def _terminal_value(self):
		last_fcff = [s.free_cash_flow + s.dividends for s in self.state_history][-1]
		term_growth_rate = 0.05
		term_discount_rate = 0.15
		term_value_undiscounted = last_fcff*12*(1 + term_growth_rate)/(term_discount_rate - term_growth_rate)
		term_value_discounted = term_value_undiscounted/(1.25**10)
		return term_value_discounted

	def _discount_rate(self):
		annual_rate = 1.25
		monthly_rate = math.pow(annual_rate, 1/12)
		return monthly_rate

	def plot(self, attribute, annualize=False):
		assert len(self.state_history) == NUM_MONTHS

		t = np.linspace(0, NUM_MONTHS - 1, NUM_MONTHS)
		a = np.array([getattr(s, attribute) for s in self.state_history])
		if annualize:
			a *= 12

		plt.figure()
		plt.plot(t, a)
		plt.title(attribute + (" (annualized)" if annualize else ""))
		plt.show()

	def timeseries(self, attribute, annualize=False):
		assert len(self.state_history) == NUM_MONTHS
		a = np.array([getattr(s, attribute) for s in self.state_history])
		if annualize:
			a *= 12
		return a


