import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Our goal is to estimate Cost of Equity for Luna (or equivalently, the discount rate
the market will apply to Luna cash flows).
According to the CAPM, Cost of Equity = Risk Free Rate + beta * Market Risk Premium
for an appropriately chosen market benchmark. We opt for the default choice of the
S&P 500. The only part of the equation that needs modeling is beta.

Given that there is no price history for Luna, it isn't possible to do a regression
on Luna vs market returns. Instead, we estimate each component of beta separately
using comparable assets. We need to estimate:
(1) corr(Luna, market)
(2) vol(Luna)
Range of comparable assets: BTC, GMO Payments (Japanese payment co), KGInicis (Korean payment co), 
Visa, Amex etc
While BTC is not equity, it can be used as a proxy for corr and vol of returns

We apply standard practice of 5-year monthly log returns data in our beta analysis.
All price data has been converted to USD at the appropriate historical rate.
"""

RISK_FREE_RATE = 0.0214 # 3-month US Treasury
MARKET_RISK_PREMIUM = 0.055 # https://assets.kpmg.com/content/dam/kpmg/nl/pdf/2018/advisory/equity-market-risk-premium-research-summary.pdf

def prices(path, attr, reverse=False):
	df = pd.read_csv(path)[[attr]]
	if reverse:
		df = df.iloc[::-1].reset_index()[[attr]] # reverse rows and re-index
	df.columns = ["price"]
	df["price"] = df["price"].astype(str).str.replace(",", "")
	df["price"] = df["price"].astype("float64")
	return df

def log_returns(prices):
	df = prices.copy() # do not write on the input DataFrame
	df.columns = ["curr"]
	df["prev"] = df.curr.shift(1)
	df = df[1:]
	df["log_ret"] = np.log(df.curr) - np.log(df.prev)
	return df[["log_ret"]]

def annualized_vol(prices):
	return log_returns(prices)["log_ret"].std()*12**0.5

def correlation(prices_1, prices_2):
	return log_returns(prices_1)["log_ret"].corr(log_returns(prices_2)["log_ret"])

if __name__ == "__main__":
	# Adj Close for the first set of data refers to the beginning of each month.
	# We use Opening price for the FX data because it refers to the entire month.
	# We reverse the FX data to match ascending order of the rest.
	SP = prices("data/^GSPC.csv", "Adj Close")
	BTC = prices("data/BTC-USD.csv", "Adj Close")
	V = prices("data/V.csv", "Adj Close")
	AXP = prices("data/AXP.csv", "Adj Close")
	GLD = prices("data/GLD.csv", "Adj Close")
	GMO = prices("data/GMO.csv", "Adj Close") # quoted in JPY
	KGI = prices("data/KGI.csv", "Adj Close") # quoted in KRW

	USD_JPY = prices("data/USD-JPY.csv", "Open", reverse=True)
	USD_KRW = prices("data/USD-KRW.csv", "Open", reverse=True)

	# we convert foreign equity data to USD according to each month's historical rate
	GMO = GMO/USD_JPY
	KGI = KGI/USD_KRW

	print("Annualized Vol")
	print("S&P 500: {}".format(annualized_vol(SP)))
	print("Visa: {}".format(annualized_vol(V)))
	print("Amex: {}".format(annualized_vol(AXP)))
	print("GMO: {}".format(annualized_vol(GMO)))
	print("KGI: {}".format(annualized_vol(KGI)))
	print("Gold: {}".format(annualized_vol(GLD)))
	print("BTC: {}".format(annualized_vol(BTC)))
	print()
	print("S&P 500 Correlation")
	print("S&P 500: {}".format(correlation(SP,SP)))
	print("Visa: {}".format(correlation(SP,V)))
	print("Amex: {}".format(correlation(SP,AXP)))
	print("GMO: {}".format(correlation(SP,GMO)))
	print("KGI: {}".format(correlation(SP,KGI)))
	print("Gold: {}".format(correlation(SP,GLD)))
	print("BTC: {}".format(correlation(SP,BTC)))
	print()
	vol_projection = (annualized_vol(BTC) + (annualized_vol(GMO) + annualized_vol(KGI))/2)/2
	corr_projection = (correlation(SP,BTC) + correlation(SP,V))/2
	beta_projection = corr_projection*vol_projection/annualized_vol(SP)
	cost_of_equity = RISK_FREE_RATE + beta_projection*MARKET_RISK_PREMIUM
	print("Luna Annualized Vol Projection: {}".format(vol_projection))
	print("Luna Market Correlation Projection: {}".format(corr_projection))
	print("Luna Beta Projection: {}".format(beta_projection))
	print("Luna Cost of Equity: {}".format(cost_of_equity))
