"""
Encapsulates the state of Terra's economy, which consists of the following:

	month
	Terra Alliance GMV
	discount
	user penetration --> TerraPay users
	volume penetration --> Terra tx volume
	seigniorage
	Terra market cap
	cash
	Target Reserve Ratio
	Fiat Reserve size
"""

class State:

	def __init__(self,
				month = -1,
				alliance_gmv = 0,
				alliance_users = 0,
				base_volume_penetration = 0,
				volume_penetration = 0,
				nominal_volume = 0,
				real_volume = 0,
				hodl_ratio = 0,
				seigniorage = 0,
				terra_commerce_market = 0,
				terra_market_cap = 0,
				target_reserve_ratio = 0,
				reserve_ratio = 0,
				target_discount = 0,
				discount = 0,
				free_cash_flow = 0,
				dividends = 0,
				cash = 0
				):
		self.month = month
		self.alliance_gmv = alliance_gmv
		self.alliance_users = alliance_users
		self.base_volume_penetration = base_volume_penetration
		self.volume_penetration = volume_penetration
		self.nominal_volume = nominal_volume
		self.real_volume = real_volume
		self.hodl_ratio = hodl_ratio
		self.seigniorage = seigniorage
		self.terra_commerce_market = terra_commerce_market
		self.terra_market_cap = terra_market_cap
		self.target_reserve_ratio = target_reserve_ratio
		self.reserve_ratio = reserve_ratio
		self.target_discount = target_discount
		self.discount = discount
		self.free_cash_flow = free_cash_flow
		self.dividends = dividends
		self.cash = cash

	def __str__(self):
		return "month {:2} \tAGMV {:6,.0f} \tBVP {:3.0%} VP {:3.0%} \t\tdiscount {:5.2%} \t\tFCFF {:5,.0f} DIV {:4,.0f} cash {:6,.0f} \tRR {:4.0%} TRR {:4.0%}".format(self.month, self.alliance_gmv, self.base_volume_penetration, self.volume_penetration, self.discount, self.free_cash_flow, self.dividends, self.cash, self.reserve_ratio, self.target_reserve_ratio)



