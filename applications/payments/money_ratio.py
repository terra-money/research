import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Historical M1 and M2 sizes for the USD, as well as near to narrow money ratio.
Recall that M1 is narrow money and M2 is narrow + near money.
"""

if __name__ == "__main__":
	M1 = pd.read_csv("data/M1.csv")
	M2 = pd.read_csv("data/M2.csv")
	M1 = M1.set_index("DATE")
	M2 = M2.set_index("DATE")
	M1.index = pd.to_datetime(M1.index)
	M2.index = pd.to_datetime(M2.index)
	M = pd.merge(M1, M2, on="DATE")
	M["NEAR/NARROW"] = M["M2"]/M["M1"] - 1

	ax = M.plot(secondary_y=["NEAR/NARROW"])
	ax.set_ylabel("$ billion")
	ax.right_ax.set_ylim([0,5])
	plt.show()