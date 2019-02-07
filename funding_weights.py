import pandas as pd
import numpy as np

LAMBDA = 2/3

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
	tv_sum = tv1 + tv2
	sm1 = spending_multiplier(tv1, df.iloc[i-1]['TV1'], df.iloc[i-1]['w1']) if i > 0 else np.NaN
	sm2 = spending_multiplier(tv2, df.iloc[i-1]['TV2'], df.iloc[i-1]['w2']) if i > 0 else np.NaN
	sm_sum = spending_multiplier_sum([sm1, sm2])

	if sm_sum == 0:
		w1 = tv1/tv_sum
		w2 = tv2/tv_sum
	else:
		w1 = LAMBDA*tv1/tv_sum + (1-LAMBDA)*sanitize(sm1)/sm_sum
		w2 = LAMBDA*tv2/tv_sum + (1-LAMBDA)*sanitize(sm2)/sm_sum

	df.iloc[i] = {'TV1': tv1, 'TV2': tv2, 'SM1': sm1, 'SM2': sm2, 'w1': w1, 'w2': w2}


if __name__ == "__main__":
	tv1 = np.concatenate([np.linspace(1,100,15), np.linspace(100,90,5)])
	tv2 = np.concatenate([np.zeros(5), np.linspace(1,50,5), np.linspace(50, 10, 10)])
	df = pd.DataFrame(data = {'TV1': tv1, 'TV2': tv2})
	df['SM1'] = np.NaN
	df['SM2'] = np.NaN
	df['w1'] = np.NaN
	df['w2'] = np.NaN

	for i in range(len(df)):
		fill_row(df, i)

	print(df)