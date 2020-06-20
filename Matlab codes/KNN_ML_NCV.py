## Imports
import numpy as np
import statsmodels
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import pystan
from sklearn.kernel_ridge import KernelRidge
import os
import xlrd
os.chdir('C:\\Users\\lakshd5\\Dropbox\\Heteroscedasticity\\Final Data')
import statsmodels
import random
from numpy import random,argsort,sqrt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure
import matplotlib as mpl
from matplotlib.collections import LineCollection
from scipy.integrate import simps
from scipy.integrate import trapz
from scipy.integrate import cumtrapz


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def kernel(distances):
    # distances is an array of size K containing distances of neighbours
    weights = 1/np.sqrt(2*np.pi) * np.exp(-distances**2/2) # Compute an array of weights however you want
    return distances

mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2.5

import warnings
warnings.filterwarnings('ignore')

import random as rnd

rnds = np.array([276,  19, 930, 228, 223, 390, 571, 142, 701, 376, 195, 844, 922,
       714, 253, 314, 721, 928, 660, 248, 622, 931, 296, 868, 687, 713,
       385, 857, 623, 828, 581,  90,  34, 152, 989, 473,  13, 756, 863,
       234, 879,  91, 736, 578, 233, 575, 240, 206,  24, 690, 920, 838,
       178, 960, 316, 575, 548, 253, 501, 309, 601, 951, 782, 584, 734,
       293, 839, 539, 587, 987, 142, 856, 493, 257,  97, 986, 504, 915,
       426, 377, 388, 185,  79, 759, 357, 556, 818, 131, 591, 588,   222,
       214, 100, 160, 344, 748,  49, 713, 737, 991])

## Data

wb = xlrd.open_workbook('RC_PD_SF1_C.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM = []
PD = []
for ii in np.arange(1,382,1):
    if sheet.cell_value(ii,0)>0 and sheet.cell_value(ii,0)<10 and sheet.cell_value(ii,1)<0.1 and sheet.cell_value(ii,1)>0:
        count = count + 1
        IM.append(sheet.cell_value(ii,0))
        PD.append(sheet.cell_value(ii,1))
        
lnIM = np.array(np.log(IM))
lnPD = np.array(np.log(PD))    

X_plot = np.log(np.linspace(0.01, 2, 100)[:, None])
X_plot_aic = (np.linspace(-4.6051701, 0.6931471, 35)[:, None])
X_plot1 = np.log(np.linspace(0.01, 2.04, 101)[:, None])

## StatsModel Fit
KRR = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'll',bw='cv_ls') # np.array([1e20])
KRR_fit = KRR.fit(X_plot)
KRR_fit_res = KRR.fit(lnIM)

KRR1 = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'lc',bw='cv_ls') # np.array([1e20])
KRR_fit1 = KRR1.fit(X_plot)
KRR_fit_res1 = KRR1.fit(lnIM)

KRR_aic = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'll',bw='aic') # np.array([1e20])
KRR_fit_aic = KRR_aic.fit(X_plot_aic)
KRR_fit_res_aic = KRR_aic.fit(lnIM)

KRR1_aic = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'lc',bw='aic') # np.array([1e20])
KRR_fit1_aic = KRR1_aic.fit(X_plot_aic)
KRR_fit_res1_aic = KRR1_aic.fit(lnIM)


## Cross validation KNN
from sklearn.model_selection import KFold

res_knn = lnPD-KRR_fit_res[0]

kf = KFold(n_splits=10)

kf.get_n_splits(res_knn)

grid = GridSearchCV(KernelDensity(kernel='gaussian'),           {'bandwidth':np.linspace(0.1,1.0,30)},cv=10)

# N_samps = 40

LogLike_RC = np.zeros(50)

count = 0

for train_index, test_index in kf.split(res_knn):
    # print("TRAIN:", train_index, "TEST:", test_index)
     
    res_knn1 = res_knn[train_index]
    
    for N_samps in np.arange(11,61,1):
        
        cou = N_samps-11
        
        
        for ind in np.arange(0,len(test_index),1):
            
            knn = neighbors.KNeighborsRegressor(N_samps, weights='uniform')
            KK = knn.fit(lnIM[train_index].reshape(-1,1),res_knn[train_index])
            KK1 = knn.kneighbors(np.array(lnIM[ind]).reshape(-1,1),N_samps)
            
            grid.fit(res_knn1[KK1[1]].reshape(-1, 1))
            
            # print(grid.best_params_['bandwidth'])
            
            kde = KernelDensity(kernel='gaussian',bandwidth=grid.best_params_['bandwidth']).fit(res_knn1[KK1[1]].reshape(-1, 1))
        
            log_dens = kde.score_samples(res_knn[ind].reshape(1, -1))
            
            LogLike_RC[cou] = LogLike_RC[cou] + log_dens
            
            count = count + 1
            
            print("RC: ", count)
            
        
        

## Data

wb = xlrd.open_workbook('SMF_PD_SF1.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM = []
PD = []
for ii in np.arange(1,382,1):
    if sheet.cell_value(ii,0)>0 and sheet.cell_value(ii,0)<10 and sheet.cell_value(ii,1)<0.1 and sheet.cell_value(ii,1)>0:
        count = count + 1
        IM.append(sheet.cell_value(ii,0))
        PD.append(sheet.cell_value(ii,1))
        
lnIM = np.array(np.log(IM))
lnPD = np.array(np.log(PD))    

X_plot = np.log(np.linspace(0.01, 2, 100)[:, None])
X_plot_aic = (np.linspace(-4.6051701, 0.6931471, 35)[:, None])
X_plot1 = np.log(np.linspace(0.01, 2.04, 101)[:, None])

## StatsModel Fit
KRR = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'll',bw='cv_ls') # np.array([1e20])
KRR_fit = KRR.fit(X_plot)
KRR_fit_res = KRR.fit(lnIM)

KRR1 = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'lc',bw='cv_ls') # np.array([1e20])
KRR_fit1 = KRR1.fit(X_plot)
KRR_fit_res1 = KRR1.fit(lnIM)

KRR_aic = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'll',bw='aic') # np.array([1e20])
KRR_fit_aic = KRR_aic.fit(X_plot_aic)
KRR_fit_res_aic = KRR_aic.fit(lnIM)

KRR1_aic = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'lc',bw='aic') # np.array([1e20])
KRR_fit1_aic = KRR1_aic.fit(X_plot_aic)
KRR_fit_res1_aic = KRR1_aic.fit(lnIM)


## Cross validation KNN
from sklearn.model_selection import KFold

res_knn = lnPD-KRR_fit_res[0]

kf = KFold(n_splits=10)

kf.get_n_splits(res_knn)

grid = GridSearchCV(KernelDensity(kernel='gaussian'),           {'bandwidth':np.linspace(0.1,1.0,30)},cv=10)

# N_samps = 40

LogLike_SMF = np.zeros(50)

count = 0

for train_index, test_index in kf.split(res_knn):
    # print("TRAIN:", train_index, "TEST:", test_index)
     
    res_knn1 = res_knn[train_index]
    
    for N_samps in np.arange(11,61,1):
        
        cou = N_samps-11
        
        
        for ind in np.arange(0,len(test_index),1):
            
            knn = neighbors.KNeighborsRegressor(N_samps, weights='uniform')
            KK = knn.fit(lnIM[train_index].reshape(-1,1),res_knn[train_index])
            KK1 = knn.kneighbors(np.array(lnIM[ind]).reshape(-1,1),N_samps)
            
            grid.fit(res_knn1[KK1[1]].reshape(-1, 1))
            
            # print(grid.best_params_['bandwidth'])
            
            kde = KernelDensity(kernel='gaussian',bandwidth=grid.best_params_['bandwidth']).fit(res_knn1[KK1[1]].reshape(-1, 1))
        
            log_dens = kde.score_samples(res_knn[ind].reshape(1, -1))
            
            LogLike_SMF[cou] = LogLike_SMF[cou] + log_dens
            
            count = count + 1
            
            print("SMF: ", count)
            
        
        
## Data

wb = xlrd.open_workbook('WOOD_PD_SF1_C.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM = []
PD = []
for ii in np.arange(1,382,1):
    if sheet.cell_value(ii,0)>0 and sheet.cell_value(ii,0)<10 and sheet.cell_value(ii,1)<0.1 and sheet.cell_value(ii,1)>0:
        count = count + 1
        IM.append(sheet.cell_value(ii,0))
        PD.append(sheet.cell_value(ii,1))
        
lnIM = np.array(np.log(IM))
lnPD = np.array(np.log(PD))    

X_plot = np.log(np.linspace(0.01, 2, 100)[:, None])
X_plot_aic = (np.linspace(-4.6051701, 0.6931471, 35)[:, None])
X_plot1 = np.log(np.linspace(0.01, 2.04, 101)[:, None])

## StatsModel Fit
KRR = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'll',bw='cv_ls') # np.array([1e20])
KRR_fit = KRR.fit(X_plot)
KRR_fit_res = KRR.fit(lnIM)

KRR1 = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'lc',bw='cv_ls') # np.array([1e20])
KRR_fit1 = KRR1.fit(X_plot)
KRR_fit_res1 = KRR1.fit(lnIM)

KRR_aic = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'll',bw='aic') # np.array([1e20])
KRR_fit_aic = KRR_aic.fit(X_plot_aic)
KRR_fit_res_aic = KRR_aic.fit(lnIM)

KRR1_aic = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'lc',bw='aic') # np.array([1e20])
KRR_fit1_aic = KRR1_aic.fit(X_plot_aic)
KRR_fit_res1_aic = KRR1_aic.fit(lnIM)


## Cross validation KNN
from sklearn.model_selection import KFold

res_knn = lnPD-KRR_fit_res[0]

kf = KFold(n_splits=10)

kf.get_n_splits(res_knn)

grid = GridSearchCV(KernelDensity(kernel='gaussian'),           {'bandwidth':np.linspace(0.1,1.0,30)},cv=10)

# N_samps = 40

LogLike_WOOD = np.zeros(50)

count = 0

for train_index, test_index in kf.split(res_knn):
    # print("TRAIN:", train_index, "TEST:", test_index)
     
    res_knn1 = res_knn[train_index]
    
    for N_samps in np.arange(11,61,1):
        
        cou = N_samps-11
        
        
        for ind in np.arange(0,len(test_index),1):
            
            knn = neighbors.KNeighborsRegressor(N_samps, weights='uniform')
            KK = knn.fit(lnIM[train_index].reshape(-1,1),res_knn[train_index])
            KK1 = knn.kneighbors(np.array(lnIM[ind]).reshape(-1,1),N_samps)
            
            grid.fit(res_knn1[KK1[1]].reshape(-1, 1))
            
            # print(grid.best_params_['bandwidth'])
            
            kde = KernelDensity(kernel='gaussian',bandwidth=grid.best_params_['bandwidth']).fit(res_knn1[KK1[1]].reshape(-1, 1))
        
            log_dens = kde.score_samples(res_knn[ind].reshape(1, -1))
            
            LogLike_WOOD[cou] = LogLike_WOOD[cou] + log_dens
            
            count = count + 1
            
            print("WOOD: ", count)
            
        
        
## Plots

# font = {'family' : 'serif',
#         'weight' : 'normal',
#         'size'   : 26}
# figure(num=None, figsize=(8, 7), facecolor='w', edgecolor='k')
# plt.rc('font', **font)
# ax = plt.gca()
# 
# 
# plt.plot(samps_space, err_rc,color = (219/255,43/255,57/255),label='RC moment frame',linewidth=4.5,lineStyle = '-')
# plt.plot(samps_space, err_smf,color = (41/255,51/255,92/255),label='Steel moment frame',linewidth=4.5,lineStyle = '--')
# plt.plot(samps_space, err_wood,color = (80/255,114/255,60/255),label='Wood shear wall',linewidth=4.5,lineStyle = ':')
# plt.plot(np.array([38,38]),np.array([0,8]),color = (0,0,0,0.5),linewidth=4.5,lineStyle = '-')
# 
# ax.scatter(samps_space[28], err_rc[28] ,color = (219/255,43/255,57/255),s=200,marker="o")
# ax.scatter(samps_space[35], err_smf[35] ,color = (41/255,51/255,92/255),s=200,marker="d")
# ax.scatter(samps_space[40], err_wood[40] ,color = (80/255,114/255,60/255),s=200,marker="s")
# 
# plt.xlabel('Number of neighbors')
# plt.ylabel('Sum of squared differences')
# plt.legend(frameon=False,loc='upper left')
# # plt.title('Cross validation')
# 
# plt.ylim([0,8])
# plt.xticks([10,15,20,25,30,35,40,45,50,55,60])
# plt.yticks([0,2,4,6,8])
# plt.show()