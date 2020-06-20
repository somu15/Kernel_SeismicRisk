## Imports
import numpy as np
import statsmodels
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import pystan

## Data
data = pd.read_csv("Data_WOOD.csv")
data_ida = pd.read_csv("WOOD_IDA.csv")
std_exact = np.zeros(38)
for x in np.arange(1,38,1):
  tmp = (data_ida['S'+str(x)])
  tmp = tmp[~np.isnan(tmp)]
  tmp = tmp[tmp<0.1]
  std_exact[x] = np.std(np.log(tmp))


## Execute
data = {'X': data.lnIM, 'Y': data.lnRD, 'N':len(data.lnIM)}
fit = pystan.stan(file='heteroscedastic.stan', data=data)

## Postprocess
print(fit)
fit.plot()

## Check results
im_req = np.arange(0.01, 1.8, 0.01)
std_req = np.exp(-1.53 - 0.12 * np.log(im_req))

plt.plot(im_req, std_req)
plt.scatter(np.arange(0, 1.9, 0.05), std_exact)

# pred = -1.55 + 1.01 * np.log(im_req)
# plt.loglog(im_req, np.exp(pred))
# plt.scatter(np.exp(data.lnIM), np.exp(data.lnRD))

 