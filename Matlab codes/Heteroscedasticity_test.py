## Imports
import numpy as np
import statsmodels
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd 

## Load Data
df = pd.read_csv("Data_WOOD.csv")
#df = df[['adjdep', 'adjfatal', 'adjsimp']]

## Perform Regression
import statsmodels.formula.api as smf
reg = smf.ols('lnRD ~ lnIM', data=df).fit()
reg.summary()

## Compute Residuals
pred_val = reg.fittedvalues.copy()
true_val = df['lnRD'].values.copy()
residual = true_val - pred_val
fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(residual, pred_val)

## Test Hetero
from statsmodels.stats.diagnostic import het_breuschpagan
_, pval, __, f_pval = het_breuschpagan(residual, df[['lnIM','lnIMsq']])
pval, f_pval