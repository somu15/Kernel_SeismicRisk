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
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF


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

import warnings
warnings.filterwarnings('ignore')

mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2.5

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

# X_plot = np.array([[np.log(0.01)],[np.log(0.1)],[np.log(0.25)],[np.log(0.4)],[np.log(0.55)],[np.log(0.7)],[np.log(0.85)],[np.log(1)],[np.log(1.15)],[np.log(1.3)],[np.log(1.45)],[np.log(1.6)],[np.log(1.75)],[np.log(1.9)]])
# X_plot_aic = (np.linspace(-4.6051701, 0.6931471, 35)[:, None])
X_plot_aic = np.log(np.linspace(0.01, 2, 100)[:, None])
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

# ## KNN function
# 
# from pylab import plot,show
# 
# def knn_search(x, D, K):
#  dis = sqrt((lnIM-x)**2)
#  idx = argsort(dis) # sorting
#  # return the indexes of K nearest neighbours
#  return idx[:K]
# 
# 
# ## Estimate standard deviation (sampling)
# 
# N_samps = 30
# 
# for ii in np.arange(0,len(X_plot),1):
#     ID = knn_search(X_plot[ii], lnIM, N_samps)
#     std_req[ii] = np.std(lnPD[ID])
# 
# plt.plot(X_plot,KRR_fit[0],X_plot,KRR_fit[0]+std_req,X_plot,KRR_fit[0]-std_req)
# plt.scatter(lnIM,lnPD)
# plt.show()

## Cross validation KNN
from sklearn.model_selection import train_test_split

res_knn = lnPD-KRR_fit_res[0]
X_train, X_test, y_train, y_test = train_test_split(lnIM, res_knn, test_size = 0.33, random_state = 1000)

std_train = np.zeros(len(X_plot))
std_test = np.zeros(len(X_plot))

samps_space = np.arange(10,51,1)
err = np.zeros(len(samps_space))

for kk in np.arange(0,len(samps_space),1):
    N_samps = samps_space[kk]

    for ind in np.arange(0,len(X_plot),1):
    
        knn = neighbors.KNeighborsRegressor(N_samps, weights='uniform')
        KK = knn.fit(X_train.reshape(-1,1),y_train)
        KK1 = knn.kneighbors(np.array(X_plot[ind]).reshape(-1,1),N_samps)
    
    # samps_rec[ind] = N_samps
        std_train[ind] = np.std(y_train[KK1[1]])
    
        KK_test = knn.fit(X_test.reshape(-1,1),y_test)
        KK1 = knn.kneighbors(np.array(X_plot[ind]).reshape(-1,1),N_samps)
    
    # samps_rec[ind] = N_samps
        std_test[ind] = np.std(y_test[KK1[1]])

    err[kk] = np.sum((std_test-std_train)*(std_test-std_train))


# print(samps_space[np.argmin(err)])

plt.plot(samps_space,err)
plt.show()

## NN Density computation

res_req = (lnPD-KRR_fit_res[0])
res_fit = statsmodels.nonparametric.kernel_regression.KernelReg(res_req, lnIM, 'c',reg_type = 'lc',bw='cv_ls')

tol = 1e-3
res_knn = lnPD-KRR_fit_res[0]
resX = np.linspace(-5, 5, 1000)[:, None]
std_est = np.zeros(len(X_plot))
samps_rec = np.zeros(len(X_plot))
comp_tol = 10

grid = GridSearchCV(KernelDensity(kernel='gaussian'),           {'bandwidth':np.linspace(0.1,1.0,30)},cv=20)

resX_plt = np.empty((len(X_plot),0))
dens_plt = np.empty((len(X_plot),0))

# X_plot = np.log(np.array([0.01, 1.6]))

N_samps = 50
for ind in np.arange(0,len(X_plot),1):# 
    # N_samps = 3
    # while comp_tol>tol:
    #     knn = neighbors.KNeighborsRegressor(N_samps, weights='uniform')
    #     KK = knn.fit(lnIM.reshape(-1,1),res_knn)
    #     KK1 = knn.kneighbors(np.array(X_plot[ind]).reshape(-1,1),N_samps)
    #     comp_tol = np.abs(np.std(res_knn[KK1[1]])-std_req[ind])
    #     N_samps = N_samps + 1
    #     print(comp_tol)
    
    knn = neighbors.KNeighborsRegressor(N_samps, weights='uniform')
    KK = knn.fit(lnIM.reshape(-1,1),res_knn)
    KK1 = knn.kneighbors(np.array(X_plot[ind]).reshape(-1,1),N_samps)
    
    # samps_rec[ind] = N_samps
    std_est[ind] = np.std(res_knn[KK1[1]])
    
    print(stats.kstest((res_knn[KK1[1]]), "norm"))
    
    
    # res_fit = statsmodels.nonparametric.kernel_regression.KernelReg(res_req, lnIM, 'c',reg_type = 'lc',bw='cv_ls')
    
    # kde = KernelDensity(kernel='gaussian',bandwidth=np.power(len(res_knn[KK1[1]]),-1/5)*1.06*np.std(res_knn[KK1[1]])).fit(res_knn[KK1[1]].reshape(-1, 1))
    # kde = KernelDensity(kernel='gaussian',bandwidth=np.power(N_samps,-1/5)*1.06*np.std(res_knn[KK1[1]])).fit(res_knn[KK1[1]].reshape(-1, 1))
    
    
    grid.fit(res_knn[KK1[1]].reshape(-1, 1))
    
    # print(grid.best_params_['bandwidth'])
    
    kde = KernelDensity(kernel='gaussian',bandwidth=grid.best_params_['bandwidth']).fit(res_knn[KK1[1]].reshape(-1, 1))

    log_dens = kde.score_samples(resX)
    
    
    
    
    if ind > 0:
        resX_plt = np.vstack((resX_plt, np.rot90(resX)))
        dens_plt = np.vstack((dens_plt, np.exp(log_dens)))
    else:
        resX_plt = np.rot90(resX)
        dens_plt = np.exp(log_dens)
        

    # plt.plot(resX,np.exp(log_dens))

# plt.show()

fig, ax = plt.subplots()
# yint = np.array([0.01,0.1,0.25,0.4,0.55,0.7,0.85,1,1.15,1.3,1.45,1.6,1.75,1.9]) # 
yint = np.arange(0.05,2.0,0.067242)
lc = multiline(resX_plt, dens_plt, yint, cmap='YlOrRd', lw=4)

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)
axcb = fig.colorbar(lc)
axcb.set_label('Sa(1.5s) [g]')
ax.set_title('RC moment frame')
plt.xlabel('Residual')
plt.ylabel('Density')

# kde1 = KernelDensity(kernel='gaussian',bandwidth=np.power(len(IM),-1/5)*1.06*np.std(res_knn)).fit(res_knn.reshape(-1, 1))
grid = GridSearchCV(KernelDensity(kernel='gaussian'),{'bandwidth':np.linspace(0.1,1.0,30)},cv=20)
grid.fit(res_knn.reshape(-1, 1))

kde1 = KernelDensity(kernel='gaussian',bandwidth=grid.best_params_['bandwidth']).fit(res_knn.reshape(-1, 1))
log_dens1 = kde1.score_samples(resX)


KRRC = statsmodels.nonparametric.kernel_regression.KernelReg((lnPD), lnIM, 'c',reg_type = 'll',bw=np.array([1000]))
KRR_C_res = KRRC.fit(lnIM)
resCl = lnPD-KRR_C_res[0]
stdC = np.std(resCl)


plt.plot(resX,norm.pdf(resX,0,stdC),color='k',label='Case 1: Linear regression',linewidth=5.0,lineStyle = '--')
plt.plot(resX,np.exp(log_dens1),color='k',label='Case 2: Gaussian kernel',linewidth=5.0,lineStyle = '-')
plt.legend(frameon=False,loc='upper left')
plt.yticks([0,0.5,1,1.5,2])
plt.show()

# WOOD: YlGn
# SMF: winter
# RCMF: YlOrRd

data = np.random.normal(0,5, size=2000)

# ecdf = ECDF(np.array(res_knn[KK1[1]])) # res_knn[KK1[1]].reshape(-1, 1)
# plt.plot(ecdf.x,ecdf.y)


## STD COMP
plt.plot(np.exp(X_plot),std_est,np.exp(X_plot),std_req)
plt.show()
    
    
    
    
    

## FIT plot
plt.plot(X_plot,KRR_fit[0],X_plot,KRR_fit[0]+std_req,X_plot,KRR_fit[0]-std_req)
plt.scatter(lnIM,lnPD)
plt.show()


## Residuals distribution

resX = np.linspace(-3, 3, 100)[:, None]

kde = KernelDensity(kernel='gaussian', bandwidth=res_fit.bw).fit(np.array(lnPD-KRR_fit_res[0]).reshape(-1, 1))

log_dens = kde.score_samples(resX)

plt.plot(resX,np.exp(log_dens))
plt.show()

## Cloud
KRRC = statsmodels.nonparametric.kernel_regression.KernelReg((lnPD), lnIM, 'c',reg_type = 'll',bw=np.array([1000]))
KRR_C_res = KRRC.fit(lnIM)
resCl = lnPD-KRR_C_res[0]
stdC = np.std(resCl)

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}
figure(num=None, figsize=(8, 7), facecolor='w', edgecolor='k')
plt.rc('font', **font)
ax = plt.gca()
# ax.scatter(IM,PD,color = (219/255,43/255,57/255,0.5),s=200,marker="o")

ax.plot(np.exp(X_plot),np.exp(KRRC.fit(X_plot)[0]),color = 'k',label='Linear regression',linewidth=3.5,lineStyle = '--')
ax.plot(np.exp(X_plot),np.exp(KRR_fit[0]),color = 'k',label='Local linear (LS-CV)',linewidth=3.5,lineStyle = '-')

# ax.plot(np.exp(X_plot),np.exp(KRR_fit1[0]),color = 'k',label='Local constant (LS-CV)',linewidth=3.5,lineStyle = '-.')
# ax.plot(np.exp(X_plot_aic),np.exp(KRR_fit_aic[0]),color = 'k',label='Local linear (AIC)',linewidth=1.0,marker = 'o',lineStyle = '-',markersize=8)
# ax.plot(np.exp(X_plot_aic),np.exp(KRR_fit1_aic[0]),color = 'k',label='Local constant (AIC)',linewidth=1.0,marker = 's',lineStyle = '-',markersize=8)

res = [] 
for idx in range(0, len(lnIM)) : 
    if lnIM[idx] > np.log(0.45): 
        res.append(idx) 

ax.plot(np.array([0.45,0.45]),np.array([0.001,0.1]),color = (0,0,0,0.5),linewidth=4.5,lineStyle = '-')

plt.scatter(np.exp(lnIM[res]),np.exp(lnPD[res]),color = (41/255,51/255,92/255,0.5),s=200,marker="d")

im = np.linspace(0.45, 2, 100)
pred = np.exp(0.7362*(np.log(np.linspace(0.45, 2, 100))-np.log(0.45))+np.log(0.02))

ax.plot(im,pred,color = (255/255,165/255,0/255),label='Linear regression (>0.45g)',linewidth=5.5,lineStyle = ':')

# ax.plot(np.exp(np.log(np.linspace(0.45, 2, 100)),np.exp(0.7362*log(np.linspace(0.45, 2, 100)+np.log(0.02)),color = 'k',label='Linear regression',linewidth=3.5,lineStyle = ':')

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Sa(1.33s) [g]')
plt.ylabel('Peak Interstory Drift')
plt.title('Steel moment frame')
plt.xticks([0.01,0.1,1])
plt.legend(frameon=False,loc='lower right')
plt.xlim([0.03,2])
plt.ylim([0.001,0.1])
plt.show()

# SMF: beta = 0.7362; 0.02 (0.45g)
# RC: beta = 0.9753; 0.0315; (0.6g)

## Error squared
res_req = (lnPD-KRR_fit_res[0])**2
res_req1 = (lnPD-KRR_fit_res1[0])**2
res_req_aic = (lnPD-KRR_fit_res_aic[0])**2
res_req1_aic = (lnPD-KRR_fit_res1_aic[0])**2

res_fit = statsmodels.nonparametric.kernel_regression.KernelReg(res_req, lnIM, 'c',reg_type = 'll',bw='cv_ls')
res_fit1 = statsmodels.nonparametric.kernel_regression.KernelReg(res_req, lnIM, 'c',reg_type = 'lc',bw='cv_ls')
res_fit_aic = statsmodels.nonparametric.kernel_regression.KernelReg(res_req_aic, lnIM, 'c',reg_type = 'll',bw='aic')
res_fit1_aic = statsmodels.nonparametric.kernel_regression.KernelReg(res_req_aic, lnIM, 'c',reg_type = 'lc',bw='aic')

res_pred = res_fit.fit(X_plot)
res_pred1 = res_fit1.fit(X_plot)
res_pred_aic = res_fit_aic.fit(X_plot_aic)
res_pred1_aic = res_fit1_aic.fit(X_plot_aic)

std_req = sqrt(res_pred[0])
std_req1 = sqrt(res_pred1[0])
std_req_aic = sqrt(res_pred_aic[0])
std_req1_aic = sqrt(res_pred1_aic[0])

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}
figure(num=None, figsize=(8, 7), facecolor='w', edgecolor='k')
plt.rc('font', **font)
ax = plt.gca()

plt.scatter(np.exp(lnIM),res_req,color = (219/255,43/255,57/255,0.5),s=200,marker="o")

plt.plot(np.exp(X_plot),res_pred[0],color = 'k',linewidth=3.0,lineStyle = '-',label='Local linear (LS-CV)')

plt.plot(np.exp(X_plot),res_pred1[0],color = 'k',linewidth=3.0,lineStyle = '-.',label='Local constant (LS-CV)')

plt.plot(np.exp(X_plot_aic),res_pred_aic[0],color = 'k',label='Local linear (AIC)',linewidth=1.0,marker = 'o',lineStyle = '-',markersize=8)

plt.plot(np.exp(X_plot_aic),res_pred1_aic[0],color = 'k',label='Local constant (AIC)',linewidth=1.0,marker = 's',lineStyle = '-',markersize=8)


# plt.legend()
plt.xlabel('Sa(1.5s) [g]')
plt.ylabel('Squared error')
plt.title('RC moment frame')
plt.legend(frameon=False)
ax = plt.gca()
ax.set_xscale('log')
plt.ylim([0,1])
plt.xlim([0.03,2])
plt.show()


## Standard deviation comparison

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

plt.plot(np.array([np.exp(X_plot[0]), np.exp(X_plot[99])]), np.array([stdC,stdC]),color = (80/255,114/255,60/255),lineStyle = 'solid',linewidth=5.0,label = 'Linear regression')
# plt.plot(np.exp(X_plot),std_req,color = (80/255,114/255,60/255),lineStyle = 'dashed',linewidth=5.0,label = 'Local linear')
plt.plot(np.exp(X_plot_aic),std_req1_aic,color = (80/255,114/255,60/255),lineStyle = 'dotted',linewidth=5.0,label = 'Method 1: Local constant (AIC)')
plt.plot(np.exp(X_plot),std_est,color = (80/255,114/255,60/255),lineStyle = 'dashdot',linewidth=5.0,label = 'Method 2: K nearest neighbor')

# poly = np.polyfit(X_plot[:,0],std_est,12)
# poly_y = np.poly1d(poly)(X_plot)
# plt.plot(np.exp(X_plot),poly_y,color = (0,0,0),lineStyle = '-',linewidth=5.0,label = 'KNN (Smoothed)')

ax = plt.gca()
ax.set_xscale('log')
plt.legend(frameon=False,loc='lower left')
plt.title('Wood shear wall')
plt.xlabel('Sa(0.53s) [g]')
plt.ylabel('Standard deviation')
plt.yticks([0.15,0.25,0.35,0.45,0.55,0.65])
plt.xlim([0.05, 2])
plt.ylim([0.1,0.65])
plt.show()

# std_est = poly_y[:,0]

## Compute fragility

# Dr = np.log(0.053)
# 
# prob1 = np.zeros(len(X_plot))
# prob2 = np.zeros(len(X_plot))
# prob3 = np.zeros(len(X_plot))
# prob4 = np.zeros(len(X_plot))
# prob5 = np.zeros(len(X_plot))
# 
# pdf1 = np.zeros(len(X_plot))
# pdf2 = np.zeros(len(X_plot))
# 
# 
# for ii in np.arange(0,len(X_plot),1):
#     fit_valC = KRRC.fit(X_plot[ii])
#     resN_Cl = (Dr-fit_valC[0])/stdC
#     prob1[ii] = 1-norm.cdf(resN_Cl)
#     
#     pdf1[ii] = norm.pdf(resN_Cl)
# 
#     
#     
#     fit_val = KRR.fit(X_plot[ii])
#     resN2 = (Dr-fit_val[0])/stdC#std_req[ii]
#     resN3 = (Dr-fit_val[0])/std_req1_aic[ii]
#     prob2[ii] = 1-norm.cdf(resN2)
#     prob3[ii] = 1-norm.cdf(resN3)
#     
#     res4 = (Dr-fit_val[0])
#     ind_req = np.argmin(abs(res4-resX))
#     prob4[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#     pdf2[ii] = np.exp(log_dens1[ind_req])
#     tmp = dens_plt[ii]
#     prob5[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
#     
#     
#     # samps_rec[ind] = N_samps
#     # std_est[ind] = np.std(res_knn[KK1[1]])
#     
# font = {'family' : 'serif',
#         'weight' : 'normal',
#         'size'   : 26}
# 
# plt.rc('font', **font)
# 
# #prob_1_H = prob1
# #prob_5_H = prob5
# 
# # plt.plot(np.exp(X_plot),prob1,color = (219/255,43/255,57/255,0.3),linewidth=4.0,label = 'Case 1',marker='o')
# plt.plot(np.exp(X_plot),prob1,color = (41/255,51/255,92/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 1')
# # 
# # plt.plot(np.exp(X_plot),prob2,color = (219/255,43/255,57/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
# plt.plot(np.exp(X_plot),prob3,color = (41/255,51/255,92/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 2') # 
# plt.plot(np.exp(X_plot),prob4,color = (41/255,51/255,92/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 3')
# # 
# plt.plot(np.exp(X_plot),prob5,color = (41/255,51/255,92/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 4')
# # 
# 
# # plt.plot(np.exp(X_plot),prob_1_H,color = (0,0,0,1),lineStyle = 'dashed',linewidth=5.0,label = 'Case 1 (High code)')
# # 
# # plt.plot(np.exp(X_plot),prob_5_H,color = (0,0,0,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 4 (High code)')
# 
# plt.legend(frameon=False,loc='lower right')
# plt.title('Steel moment frame (Moderate)')
# plt.xlabel('Sa(1.33s) [g]')
# plt.ylabel('Exceedance probability')
# plt.ylim((0, 1))
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.show()
    


Dr = np.log(0.1)
Dr1 = np.log(0.0533)

prob1 = np.zeros(len(X_plot))
prob2 = np.zeros(len(X_plot))
prob3 = np.zeros(len(X_plot))
prob4 = np.zeros(len(X_plot))
prob5 = np.zeros(len(X_plot))

prob11 = np.zeros(len(X_plot))
prob21 = np.zeros(len(X_plot))
prob31 = np.zeros(len(X_plot))
prob41 = np.zeros(len(X_plot))
prob51 = np.zeros(len(X_plot))


pdf1 = np.zeros(len(X_plot))
pdf2 = np.zeros(len(X_plot))


for ii in np.arange(0,len(X_plot),1):
    
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr-fit_valC[0])/stdC
    prob1[ii] = 1-norm.cdf(resN_Cl)
    
    
    
    
    fit_val = KRR.fit(X_plot[ii])
    res4 = (Dr-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob2[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    
    resN3 = (Dr-fit_val[0])/std_req1_aic[ii]
    prob3[ii] = 1-norm.cdf(resN3)
    
    resN4 = (Dr-fit_val[0])/std_est[ii]
    prob4[ii] = 1-norm.cdf(resN4)
    
    tmp = dens_plt[ii]
    prob5[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    
    
    
    
    # fit_valC = KRRC.fit(X_plot[ii])
    # resN_Cl = (Dr1-fit_valC[0])/stdC
    # prob11[ii] = 1-norm.cdf(resN_Cl)
    # 
   
#   #   res4 = (Dr1-fit_val[0])
    # ind_req = np.argmin(abs(res4-resX))
    # prob21[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    # 
    # 
    # fit_val = KRR.fit(X_plot[ii])
    # resN3 = (Dr1-fit_val[0])/std_req1_aic[ii]
    # prob31[ii] = 1-norm.cdf(resN3)
    # 
    # resN4 = (Dr1-fit_val[0])/std_est[ii]
    # prob41[ii] = 1-norm.cdf(resN4)
    # 
    # tmp = dens_plt[ii]
    # prob51[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

#prob_1_H = prob1
#prob_5_H = prob5
prob2[0] = 0
prob5[0] = 0
# plt.plot(np.exp(X_plot),prob1,color = (219/255,43/255,57/255,0.3),linewidth=4.0,label = 'Case 1',marker='o')
# plt.plot(np.exp(X_plot),prob1,color = (219/255,43/255,57/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 1')
# 
plt.scatter(np.exp(X_plot),prob1,color = (80/255,114/255,60/255,0.3),label = 'Case 1',s=50,marker='o')# 
plt.plot(np.exp(X_plot),prob2,color = (80/255,114/255,60/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')# 
plt.plot(np.exp(X_plot),prob3,color = (80/255,114/255,60/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3') # 
plt.plot(np.exp(X_plot),prob4,color = (80/255,114/255,60/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4') # 
plt.plot(np.exp(X_plot),prob5,color = (80/255,114/255,60/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')



# plt.scatter(np.exp(X_plot),prob11,color = (0,0,0,0.3),s=50,marker='o')# ,label = 'Case 1'
# plt.plot(np.exp(X_plot),prob21,color = (0,0,0,0.5),lineStyle = 'dashed',linewidth=5.0)# ,label = 'Case 2'
# plt.plot(np.exp(X_plot),prob31,color = (0,0,0,0.7),lineStyle = 'dotted',linewidth=5.0) # ,label = 'Case 3'
# plt.plot(np.exp(X_plot),prob41,color = (0,0,0,0.85),lineStyle = 'dashdot',linewidth=5.0) # ,label = 'Case 4'
# plt.plot(np.exp(X_plot),prob51,color = (0,0,0,1),lineStyle = 'solid',linewidth=5.0,label = 'High-code')

# 

# plt.plot(np.exp(X_plot),prob_1_H,color = (0,0,0,1),lineStyle = 'dashed',linewidth=5.0,label = 'Case 1 (High code)')
# 
# plt.plot(np.exp(X_plot),prob_5_H,color = (0,0,0,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 4 (High code)')

# plt.legend(frameon=False,loc='lower right')
plt.title('Wood shear wall (Complete)')
plt.xlabel('Sa(0.53s) [g]')
plt.ylabel('Exceedance probability')
plt.ylim((0.01, 0.1))
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
plt.show()

## Compute risk

# Dr1 = np.log(0.004)
# Dr2 = np.log(0.012)
# Dr3 = np.log(0.04)
# Dr4 = np.log(0.1)
# 
# prob1_1 = np.zeros(len(X_plot))
# prob2_1 = np.zeros(len(X_plot))
# prob3_1 = np.zeros(len(X_plot))
# prob4_1 = np.zeros(len(X_plot))
# prob5_1 = np.zeros(len(X_plot))
# 
# prob1_2 = np.zeros(len(X_plot))
# prob2_2 = np.zeros(len(X_plot))
# prob3_2 = np.zeros(len(X_plot))
# prob4_2 = np.zeros(len(X_plot))
# prob5_2 = np.zeros(len(X_plot))
# 
# prob1_3 = np.zeros(len(X_plot))
# prob2_3 = np.zeros(len(X_plot))
# prob3_3 = np.zeros(len(X_plot))
# prob4_3 = np.zeros(len(X_plot))
# prob5_3 = np.zeros(len(X_plot))
# 
# prob1_4 = np.zeros(len(X_plot))
# prob2_4 = np.zeros(len(X_plot))
# prob3_4 = np.zeros(len(X_plot))
# prob4_4 = np.zeros(len(X_plot))
# prob5_4 = np.zeros(len(X_plot))
# 
# pdf1 = np.zeros(len(X_plot))
# pdf2 = np.zeros(len(X_plot))
# 
# 
# for ii in np.arange(0,len(X_plot),1):
#     fit_valC = KRRC.fit(X_plot[ii])
#     resN_Cl = (Dr1-fit_valC[0])/stdC
#     prob1_1[ii] = 1-norm.cdf(resN_Cl)
#     pdf1[ii] = norm.pdf(resN_Cl)
#     fit_val = KRR.fit(X_plot[ii])
#     resN2 = (Dr1-fit_val[0])/std_req[ii]
#     resN3 = (Dr1-fit_val[0])/std_req1[ii]
#     prob2_1[ii] = 1-norm.cdf(resN2)
#     prob3_1[ii] = 1-norm.cdf(resN3)
#     res4 = (Dr1-fit_val[0])
#     ind_req = np.argmin(abs(res4-resX))
#     prob4_1[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#     pdf2[ii] = np.exp(log_dens1[ind_req])
#     tmp = dens_plt[ii]
#     prob5_1[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
#     
#     fit_valC = KRRC.fit(X_plot[ii])
#     resN_Cl = (Dr2-fit_valC[0])/stdC
#     prob1_2[ii] = 1-norm.cdf(resN_Cl)
#     pdf1[ii] = norm.pdf(resN_Cl)
#     fit_val = KRR.fit(X_plot[ii])
#     resN2 = (Dr2-fit_val[0])/std_req[ii]
#     resN3 = (Dr2-fit_val[0])/std_req1[ii]
#     prob2_2[ii] = 1-norm.cdf(resN2)
#     prob3_2[ii] = 1-norm.cdf(resN3)
#     res4 = (Dr2-fit_val[0])
#     ind_req = np.argmin(abs(res4-resX))
#     prob4_2[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#     pdf2[ii] = np.exp(log_dens1[ind_req])
#     tmp = dens_plt[ii]
#     prob5_2[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
#     
#     fit_valC = KRRC.fit(X_plot[ii])
#     resN_Cl = (Dr3-fit_valC[0])/stdC
#     prob1_3[ii] = 1-norm.cdf(resN_Cl)
#     pdf1[ii] = norm.pdf(resN_Cl)
#     fit_val = KRR.fit(X_plot[ii])
#     resN2 = (Dr3-fit_val[0])/std_req[ii]
#     resN3 = (Dr3-fit_val[0])/std_req1[ii]
#     prob2_3[ii] = 1-norm.cdf(resN2)
#     prob3_3[ii] = 1-norm.cdf(resN3)
#     res4 = (Dr3-fit_val[0])
#     ind_req = np.argmin(abs(res4-resX))
#     prob4_3[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#     pdf2[ii] = np.exp(log_dens1[ind_req])
#     tmp = dens_plt[ii]
#     prob5_3[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
#     
#     fit_valC = KRRC.fit(X_plot[ii])
#     resN_Cl = (Dr4-fit_valC[0])/stdC
#     prob1_4[ii] = 1-norm.cdf(resN_Cl)
#     pdf1[ii] = norm.pdf(resN_Cl)
#     fit_val = KRR.fit(X_plot[ii])
#     resN2 = (Dr4-fit_val[0])/std_req[ii]
#     resN3 = (Dr4-fit_val[0])/std_req1[ii]
#     prob2_4[ii] = 1-norm.cdf(resN2)
#     prob3_4[ii] = 1-norm.cdf(resN3)
#     res4 = (Dr4-fit_val[0])
#     ind_req = np.argmin(abs(res4-resX))
#     prob4_4[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#     pdf2[ii] = np.exp(log_dens1[ind_req])
#     tmp = dens_plt[ii]
#     prob5_4[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
# 
# P1_oo = 1-prob1_1
# P1_S = prob1_1-prob1_2
# P1_M = prob1_2-prob1_3
# P1_Se = prob1_3-prob1_4
# P1_C = prob1_4
# 
# P2_oo = 1-prob2_1
# P2_S = prob2_1-prob2_2
# P2_M = prob2_2-prob2_3
# P2_Se = prob2_3-prob2_4
# P2_C = prob2_4
# 
# P3_oo = 1-prob3_1
# P3_S = prob3_1-prob3_2
# P3_M = prob3_2-prob3_3
# P3_Se = prob3_3-prob3_4
# P3_C = prob3_4
# 
# P4_oo = 1-prob4_1
# P4_S = prob4_1-prob4_2
# P4_M = prob4_2-prob4_3
# P4_Se = prob4_3-prob4_4
# P4_C = prob4_4
# 
# P5_oo = 1-prob5_1
# P5_S = prob5_1-prob5_2
# P5_M = prob5_2-prob5_3
# P5_Se = prob5_3-prob5_4
# P5_C = prob5_4


Dr1 = np.log(0.004)
Dr2 = np.log(0.008)
Dr3 = np.log(0.02)
Dr4 = np.log(0.0533)

prob1_1 = np.zeros(len(X_plot))
prob2_1 = np.zeros(len(X_plot))
prob3_1 = np.zeros(len(X_plot))
prob4_1 = np.zeros(len(X_plot))
prob5_1 = np.zeros(len(X_plot))

prob1_2 = np.zeros(len(X_plot))
prob2_2 = np.zeros(len(X_plot))
prob3_2 = np.zeros(len(X_plot))
prob4_2 = np.zeros(len(X_plot))
prob5_2 = np.zeros(len(X_plot))

prob1_3 = np.zeros(len(X_plot))
prob2_3 = np.zeros(len(X_plot))
prob3_3 = np.zeros(len(X_plot))
prob4_3 = np.zeros(len(X_plot))
prob5_3 = np.zeros(len(X_plot))

prob1_4 = np.zeros(len(X_plot))
prob2_4 = np.zeros(len(X_plot))
prob3_4 = np.zeros(len(X_plot))
prob4_4 = np.zeros(len(X_plot))
prob5_4 = np.zeros(len(X_plot))

pdf1 = np.zeros(len(X_plot))
pdf2 = np.zeros(len(X_plot))


for ii in np.arange(0,len(X_plot),1):
    
    
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr1-fit_valC[0])/stdC
    prob1_1[ii] = 1-norm.cdf(resN_Cl)
    fit_val = KRR.fit(X_plot[ii])
    res4 = (Dr1-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob2_1[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    resN3 = (Dr1-fit_val[0])/std_req1_aic[ii]
    prob3_1[ii] = 1-norm.cdf(resN3)
    resN4 = (Dr1-fit_val[0])/std_est[ii]
    prob4_1[ii] = 1-norm.cdf(resN4)
    tmp = dens_plt[ii]
    prob5_1[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr2-fit_valC[0])/stdC
    prob1_2[ii] = 1-norm.cdf(resN_Cl)
    fit_val = KRR.fit(X_plot[ii])
    res4 = (Dr2-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob2_2[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    resN3 = (Dr2-fit_val[0])/std_req1_aic[ii]
    prob3_2[ii] = 1-norm.cdf(resN3)
    resN4 = (Dr2-fit_val[0])/std_est[ii]
    prob4_2[ii] = 1-norm.cdf(resN4)
    tmp = dens_plt[ii]
    prob5_2[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr3-fit_valC[0])/stdC
    prob1_3[ii] = 1-norm.cdf(resN_Cl)
    fit_val = KRR.fit(X_plot[ii])
    res4 = (Dr3-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob2_3[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    resN3 = (Dr3-fit_val[0])/std_req1_aic[ii]
    prob3_3[ii] = 1-norm.cdf(resN3)
    resN4 = (Dr3-fit_val[0])/std_est[ii]
    prob4_3[ii] = 1-norm.cdf(resN4)
    tmp = dens_plt[ii]
    prob5_3[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr4-fit_valC[0])/stdC
    prob1_4[ii] = 1-norm.cdf(resN_Cl)
    fit_val = KRR.fit(X_plot[ii])
    res4 = (Dr4-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob2_4[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    resN3 = (Dr4-fit_val[0])/std_req1_aic[ii]
    prob3_4[ii] = 1-norm.cdf(resN3)
    resN4 = (Dr4-fit_val[0])/std_est[ii]
    prob4_4[ii] = 1-norm.cdf(resN4)
    tmp = dens_plt[ii]
    prob5_4[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    
P1_oo = 1-prob1_1
P1_S = prob1_1-prob1_2
P1_M = prob1_2-prob1_3
P1_Se = prob1_3-prob1_4
P1_C = prob1_4

P2_oo = 1-prob2_1
P2_S = prob2_1-prob2_2
P2_M = prob2_2-prob2_3
P2_Se = prob2_3-prob2_4
P2_C = prob2_4

P3_oo = 1-prob3_1
P3_S = prob3_1-prob3_2
P3_M = prob3_2-prob3_3
P3_Se = prob3_3-prob3_4
P3_C = prob3_4

P4_oo = 1-prob4_1
P4_S = prob4_1-prob4_2
P4_M = prob4_2-prob4_3
P4_Se = prob4_3-prob4_4
P4_C = prob4_4

P5_oo = 1-prob5_1
P5_S = prob5_1-prob5_2
P5_M = prob5_2-prob5_3
P5_Se = prob5_3-prob5_4
P5_C = prob5_4

## Seismic hazard

wb = xlrd.open_workbook('Sa053_LA.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM_haz = []
Haz = []
for ii in np.arange(1,51,1):
    count = count + 1
    IM_haz.append(sheet.cell_value(ii,0))
    Haz.append(sheet.cell_value(ii,1))

int_fun = interp1d(np.log(IM_haz),np.log(Haz))
Haz_int = np.exp(int_fun(X_plot1))

Haz_diff = np.abs(np.array([Haz_int[i + 1] - Haz_int[i] for i in range(len(Haz_int)-1)]))

## Compute demand hazard

Dr_req = np.log(np.linspace(0.001,0.06,100))

prob1_1 = np.zeros(len(X_plot))
prob2_1 = np.zeros(len(X_plot))
prob3_1 = np.zeros(len(X_plot))
prob4_1 = np.zeros(len(X_plot))
prob5_1 = np.zeros(len(X_plot))

DemHaz1 = np.zeros(len(Dr_req))
DemHaz2 = np.zeros(len(Dr_req))
DemHaz3 = np.zeros(len(Dr_req))
DemHaz4 = np.zeros(len(Dr_req))
DemHaz5 = np.zeros(len(Dr_req))

for ii in np.arange(0,len(Dr_req),1):
    Dr1 = Dr_req[ii]
    prob1_1 = np.zeros(len(X_plot))
    prob2_1 = np.zeros(len(X_plot))
    prob3_1 = np.zeros(len(X_plot))
    prob4_1 = np.zeros(len(X_plot))
    prob5_1 = np.zeros(len(X_plot))
    for jj in np.arange(0,len(X_plot),1):
        
        fit_valC = KRRC.fit(X_plot[jj])
        resN_Cl = (Dr1-fit_valC[0])/stdC
        prob1_1[jj] = 1-norm.cdf(resN_Cl)
        fit_val = KRR.fit(X_plot[jj])
        res4 = (Dr1-fit_val[0])
        ind_req = np.argmin(abs(res4-resX))
        prob2_1[jj] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
        resN3 = (Dr1-fit_val[0])/std_req1_aic[jj]
        prob3_1[jj] = 1-norm.cdf(resN3)
        resN4 = (Dr1-fit_val[0])/std_est[jj]
        prob4_1[jj] = 1-norm.cdf(resN4)
        tmp = dens_plt[jj]
        prob5_1[jj] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
        
    DemHaz1[ii] = np.sum(prob1_1 * np.rot90(Haz_diff))
    DemHaz2[ii] = np.sum(prob2_1 * np.rot90(Haz_diff))
    DemHaz3[ii] = np.sum(prob3_1 * np.rot90(Haz_diff))
    DemHaz4[ii] = np.sum(prob4_1 * np.rot90(Haz_diff))
    DemHaz5[ii] = np.sum(prob5_1 * np.rot90(Haz_diff))

ax = plt.gca()
# from matplotlib.pyplot import figure
# figure(num=None, figsize=(8, 6))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

plt.scatter(np.exp(Dr_req),DemHaz1,color = (80/255,114/255,60/255,0.3),label = 'Case 1',s=50,marker='o')
plt.plot(np.exp(Dr_req),DemHaz2,color = (80/255,114/255,60/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
plt.plot(np.exp(Dr_req),DemHaz3,color = (80/255,114/255,60/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
plt.plot(np.exp(Dr_req),DemHaz4,color = (80/255,114/255,60/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
plt.plot(np.exp(Dr_req),DemHaz5,color = (80/255,114/255,60/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')

plt.legend(frameon=False,loc='lower left')
plt.title('Wood shear wall')
plt.xlabel('Peak Interstory Drift')
plt.ylabel('Exceedance probability in 50 years')
# plt.ylim((0, 1))

ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

## Dem haz deaggregation

Dr_req = np.log(0.05)

prob1_1 = np.zeros(len(X_plot))
prob2_1 = np.zeros(len(X_plot))
prob3_1 = np.zeros(len(X_plot))
prob4_1 = np.zeros(len(X_plot))
prob5_1 = np.zeros(len(X_plot))

DemHaz1 = np.zeros(len(X_plot))
DemHaz2 = np.zeros(len(X_plot))
DemHaz3 = np.zeros(len(X_plot))
DemHaz4 = np.zeros(len(X_plot))
DemHaz5 = np.zeros(len(X_plot))

Dr1 = Dr_req
for jj in np.arange(0,len(X_plot),1):
    
    fit_valC = KRRC.fit(X_plot[jj])
    resN_Cl = (Dr1-fit_valC[0])/stdC
    prob1_1[jj] = 1-norm.cdf(resN_Cl)
    fit_val = KRR.fit(X_plot[jj])
    res4 = (Dr1-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob2_1[jj] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    resN3 = (Dr1-fit_val[0])/std_req1_aic[jj]
    prob3_1[jj] = 1-norm.cdf(resN3)
    resN4 = (Dr1-fit_val[0])/std_est[jj]
    prob4_1[jj] = 1-norm.cdf(resN4)
    tmp = dens_plt[jj]
    prob5_1[jj] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
        
DemHaz1 = (prob1_1 * np.rot90(Haz_diff))
DemHaz1 = DemHaz1/(np.sum(DemHaz1)*0.0201)
DemHaz2 = (prob2_1 * np.rot90(Haz_diff))
DemHaz2 = DemHaz2/(np.sum(DemHaz2)*0.0201)
DemHaz3 = (prob3_1 * np.rot90(Haz_diff))
DemHaz3 = DemHaz3/(np.sum(DemHaz3)*0.0201)
DemHaz4 = (prob4_1 * np.rot90(Haz_diff))
DemHaz4 = DemHaz4/(np.sum(DemHaz4)*0.0201)
DemHaz5 = (prob5_1 * np.rot90(Haz_diff))
DemHaz5 = DemHaz5/(np.sum(DemHaz5)*0.0201)

ax = plt.gca()
# from matplotlib.pyplot import figure
# figure(num=None, figsize=(8, 6))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

plt.scatter(np.exp(X_plot),np.rot90(DemHaz1),color = (219/255,43/255,57/255,0.3),label = 'Case 1',s=50,marker='o')
plt.plot(np.exp(X_plot),np.rot90(DemHaz2),color = (219/255,43/255,57/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
plt.plot(np.exp(X_plot),np.rot90(DemHaz3),color = (219/255,43/255,57/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
plt.plot(np.exp(X_plot),np.rot90(DemHaz4),color = (219/255,43/255,57/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
plt.plot(np.exp(X_plot),np.rot90(DemHaz5),color = (219/255,43/255,57/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')

plt.legend(frameon=False,loc='upper left')
plt.title('RC moment frame')
plt.xlabel('Sa(1.5s) g')
plt.ylabel('Density')
# plt.ylim((0, 0.04))

plt.show()

# plt.plot(time_req,time_haz1,color = (219/255,43/255,57/255,0.3),lineStyle = (0,(3,5,1,5,1,5)),linewidth=4.0,label = 'Case 1')
# # plt.plot(time_req,time_haz1,color = (80/255,114/255,60/255,0.4),lineStyle = 'dashed',linewidth=5.0,label = 'Case 1')
# plt.plot(time_req,time_haz2,color = (219/255,43/255,57/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
# plt.plot(time_req,time_haz3,color = (219/255,43/255,57/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
# plt.plot(time_req,time_haz4,color = (219/255,43/255,57/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
# plt.plot(time_req,time_haz5,color = (219/255,43/255,57/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')


# plt.scatter(time_req,time_haz1,color = (41/255,51/255,92/255,0.3),label = 'Case 1',s=50,marker='o')# 
# plt.plot(time_req,time_haz2,color = (41/255,51/255,92/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')# 
# plt.plot(time_req,time_haz3,color = (41/255,51/255,92/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3') # 
# plt.plot(time_req,time_haz4,color = (41/255,51/255,92/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4') # 
# plt.plot(time_req,time_haz5,color = (41/255,51/255,92/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')


## DS conditional upon IM to loss

Dr1 = np.log(0.004)
Dr2 = np.log(0.012)
Dr3 = np.log(0.04)
Dr4 = np.log(0.1)

std_req = 0.5

Dr_req = np.log(np.linspace(0.001,0.06,101))
Dr_req1 = np.log(np.linspace(0.001,0.06,100))

prob1_1 = np.zeros(len(Dr_req))
prob2_1 = np.zeros(len(Dr_req))
prob3_1 = np.zeros(len(Dr_req))
prob4_1 = np.zeros(len(Dr_req))
prob5_1 = np.zeros(len(Dr_req))

prob_dm1_1 = np.zeros(len(X_plot))
prob_dm1_2 = np.zeros(len(X_plot))
prob_dm1_3 = np.zeros(len(X_plot))
prob_dm1_4 = np.zeros(len(X_plot))
prob_dm1_5 = np.zeros(len(X_plot))

prob_dm2_1 = np.zeros(len(X_plot))
prob_dm2_2 = np.zeros(len(X_plot))
prob_dm2_3 = np.zeros(len(X_plot))
prob_dm2_4 = np.zeros(len(X_plot))
prob_dm2_5 = np.zeros(len(X_plot))

prob_dm3_1 = np.zeros(len(X_plot))
prob_dm3_2 = np.zeros(len(X_plot))
prob_dm3_3 = np.zeros(len(X_plot))
prob_dm3_4 = np.zeros(len(X_plot))
prob_dm3_5 = np.zeros(len(X_plot))

prob_dm4_1 = np.zeros(len(X_plot))
prob_dm4_2 = np.zeros(len(X_plot))
prob_dm4_3 = np.zeros(len(X_plot))
prob_dm4_4 = np.zeros(len(X_plot))
prob_dm4_5 = np.zeros(len(X_plot))

# prob_dm1 = np.zeros(len(Dr_req1))
# prob_dm2 = np.zeros(len(Dr_req1))
# prob_dm3 = np.zeros(len(Dr_req1))
# prob_dm4 = np.zeros(len(Dr_req1))
# prob_dm5 = np.zeros(len(Dr_req1))

# ii = 50
# for kk in np.arange(0,len(Dr_req),1):
#     Dr1 = Dr_req[kk]
#     fit_valC = KRRC.fit(X_plot[ii])
#     resN_Cl = (Dr1-fit_valC[0])/stdC
#     prob1_1[kk] = 1-norm.cdf(resN_Cl)
#     fit_val = KRR.fit(X_plot[ii])
#     res4 = (Dr1-fit_val[0])
#     ind_req = np.argmin(abs(res4-resX))
#     prob2_1[kk] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#     resN3 = (Dr1-fit_val[0])/std_req1_aic[ii]
#     prob3_1[kk] = 1-norm.cdf(resN3)
#     resN4 = (Dr1-fit_val[0])/std_est[ii]
#     prob4_1[kk] = 1-norm.cdf(resN4)
#     tmp = dens_plt[ii]
#     prob5_1[kk] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
# 
# Test = np.abs(np.array([prob1_1[i + 1] - prob1_1[i] for i in range(len(prob1_1)-1)]))
# 
# Test = Test/(np.sum(Test)*0.00059596) # *(1-np.min(prob1_1))

for ii in np.arange(0,len(X_plot),1):
    
    for kk in np.arange(0,len(Dr_req),1):
        
        Dr1 = Dr_req[kk]
        fit_valC = KRRC.fit(X_plot[ii])
        resN_Cl = (Dr1-fit_valC[0])/stdC
        prob1_1[kk] = 1-norm.cdf(resN_Cl)
        fit_val = KRR.fit(X_plot[ii])
        res4 = (Dr1-fit_val[0])
        ind_req = np.argmin(abs(res4-resX))
        prob2_1[kk] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
        resN3 = (Dr1-fit_val[0])/std_req1_aic[ii]
        prob3_1[kk] = 1-norm.cdf(resN3)
        resN4 = (Dr1-fit_val[0])/std_est[ii]
        prob4_1[kk] = 1-norm.cdf(resN4)
        tmp = dens_plt[ii]
        prob5_1[kk] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
       
    Test1 = np.abs(np.array([prob1_1[i + 1] - prob1_1[i] for i in range(len(prob1_1)-1)]))
    Test1 = Test1/(np.sum(Test1)*0.00059596) # *(1-np.min(prob1_1))
    
    Test2 = np.abs(np.array([prob2_1[i + 1] - prob2_1[i] for i in range(len(prob2_1)-1)]))
    Test2 = Test2/(np.sum(Test2)*0.00059596) # *(1-np.min(prob1_1))
    
    Test3 = np.abs(np.array([prob3_1[i + 1] - prob3_1[i] for i in range(len(prob3_1)-1)]))
    Test3 = Test3/(np.sum(Test3)*0.00059596) # *(1-np.min(prob1_1))
    
    Test4 = np.abs(np.array([prob4_1[i + 1] - prob4_1[i] for i in range(len(prob4_1)-1)]))
    Test4 = Test4/(np.sum(Test4)*0.00059596) # *(1-np.min(prob1_1))
    
    Test5 = np.abs(np.array([prob5_1[i + 1] - prob5_1[i] for i in range(len(prob5_1)-1)]))
    Test5 = Test5/(np.sum(Test5)*0.00059596) # *(1-np.min(prob1_1))
    
    prob_dm1_1[ii] = np.sum(norm.pdf((Dr_req1-Dr1)/std_req) * Test1)*0.00059596
    prob_dm1_2[ii] = np.sum(norm.pdf((Dr_req1-Dr1)/std_req) * Test2)*0.00059596
    prob_dm1_3[ii] = np.sum(norm.pdf((Dr_req1-Dr1)/std_req) * Test3)*0.00059596
    prob_dm1_4[ii] = np.sum(norm.pdf((Dr_req1-Dr1)/std_req) * Test4)*0.00059596
    prob_dm1_5[ii] = np.sum(norm.pdf((Dr_req1-Dr1)/std_req) * Test5)*0.00059596
     
    prob_dm2_1[ii] = np.sum(norm.pdf((Dr_req1-Dr2)/std_req) * Test1)*0.00059596
    prob_dm2_2[ii] = np.sum(norm.pdf((Dr_req1-Dr2)/std_req) * Test2)*0.00059596
    prob_dm2_3[ii] = np.sum(norm.pdf((Dr_req1-Dr2)/std_req) * Test3)*0.00059596
    prob_dm2_4[ii] = np.sum(norm.pdf((Dr_req1-Dr2)/std_req) * Test4)*0.00059596
    prob_dm2_5[ii] = np.sum(norm.pdf((Dr_req1-Dr2)/std_req) * Test5)*0.00059596
     
    prob_dm3_1[ii] = np.sum(norm.pdf((Dr_req1-Dr3)/std_req) * Test1)*0.00059596
    prob_dm3_2[ii] = np.sum(norm.pdf((Dr_req1-Dr3)/std_req) * Test2)*0.00059596
    prob_dm3_3[ii] = np.sum(norm.pdf((Dr_req1-Dr3)/std_req) * Test3)*0.00059596
    prob_dm3_4[ii] = np.sum(norm.pdf((Dr_req1-Dr3)/std_req) * Test4)*0.00059596
    prob_dm3_5[ii] = np.sum(norm.pdf((Dr_req1-Dr3)/std_req) * Test5)*0.00059596
    
    prob_dm4_1[ii] = np.sum(norm.pdf((Dr_req1-Dr4)/std_req) * Test1)*0.00059596
    prob_dm4_2[ii] = np.sum(norm.pdf((Dr_req1-Dr4)/std_req) * Test2)*0.00059596
    prob_dm4_3[ii] = np.sum(norm.pdf((Dr_req1-Dr4)/std_req) * Test3)*0.00059596
    prob_dm4_4[ii] = np.sum(norm.pdf((Dr_req1-Dr4)/std_req) * Test4)*0.00059596
    prob_dm4_5[ii] = np.sum(norm.pdf((Dr_req1-Dr4)/std_req) * Test5)*0.00059596
    
time_1 = 20
time_2 = 90
time_3 = 360
time_4 = 480

med_times = np.array([time_1,time_2,time_3,time_4])

std_rec = 0.75

time_req =  (np.linspace(10, 2000, 200)[:, None])

time_haz1 = np.zeros(len(time_req))
time_haz2 = np.zeros(len(time_req))
time_haz3 = np.zeros(len(time_req))
time_haz4 = np.zeros(len(time_req))
time_haz5 = np.zeros(len(time_req))

for ii in np.arange(0,len(time_req),1):
    
    time_im1 = np.zeros(len(X_plot))
    time_im2 = np.zeros(len(X_plot))
    time_im3 = np.zeros(len(X_plot))
    time_im4 = np.zeros(len(X_plot))
    time_im5 = np.zeros(len(X_plot))

    for kk in np.arange(0,len(X_plot),1):
        time_im1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * prob_dm1_1[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * prob_dm2_1[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * prob_dm3_1[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * prob_dm4_1[kk]
        
        
        time_im2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * prob_dm1_2[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * prob_dm2_2[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * prob_dm3_2[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * prob_dm4_2[kk]
        
        time_im3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * prob_dm1_3[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * prob_dm2_3[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * prob_dm3_3[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * prob_dm4_3[kk]
        
        time_im4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * prob_dm1_4[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * prob_dm2_4[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * prob_dm3_4[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * prob_dm4_4[kk]
        
        time_im5[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * prob_dm1_5[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * prob_dm2_5[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * prob_dm3_5[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * prob_dm4_5[kk]
        
    time_haz1[ii] = np.sum(time_im1 * np.rot90(Haz_diff))
    time_haz2[ii] = np.sum(time_im2 * np.rot90(Haz_diff))
    time_haz3[ii] = np.sum(time_im3 * np.rot90(Haz_diff))
    time_haz4[ii] = np.sum(time_im4 * np.rot90(Haz_diff))
    time_haz5[ii] = np.sum(time_im5 * np.rot90(Haz_diff))
        
ax = plt.gca()

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

# plt.plot(time_req,time_haz1,color = (219/255,43/255,57/255,0.3),lineStyle = (0,(3,5,1,5,1,5)),linewidth=4.0,label = 'Case 1')
# # plt.plot(time_req,time_haz1,color = (80/255,114/255,60/255,0.4),lineStyle = 'dashed',linewidth=5.0,label = 'Case 1')
# plt.plot(time_req,time_haz2,color = (219/255,43/255,57/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
# plt.plot(time_req,time_haz3,color = (219/255,43/255,57/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
# plt.plot(time_req,time_haz4,color = (219/255,43/255,57/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
# plt.plot(time_req,time_haz5,color = (219/255,43/255,57/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')


plt.scatter(time_req,time_haz1,color = (80/255,114/255,60/255,0.3),label = 'Case 1',s=50,marker='o')# 
plt.plot(time_req,time_haz2,color = (80/255,114/255,60/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')# 
plt.plot(time_req,time_haz3,color = (80/255,114/255,60/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3') # 
plt.plot(time_req,time_haz4,color = (80/255,114/255,60/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4') # 
plt.plot(time_req,time_haz5,color = (80/255,114/255,60/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')


plt.legend(frameon=False,loc='lower left')
plt.title('Wood shear wall')
plt.xlabel('Time [days]')
plt.ylabel('Exceedance probability in 50 years')
# plt.ylim((0, 1))

ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
    

# for ii in np.arange(0,len(X_plot),1):
#     
#     for jj in np.arange(0,len(Dr_req),1):
#         
#         for kk in np.arange(0,len(Dr_req),1):
#             
#             Dr1 = Dr_req[kk]
#             
#             fit_valC = KRRC.fit(X_plot[ii])
#             resN_Cl = (Dr1-fit_valC[0])/stdC
#             prob1_1[kk] = 1-norm.cdf(resN_Cl)
#             fit_val = KRR.fit(X_plot[ii])
#             res4 = (Dr1-fit_val[0])
#             ind_req = np.argmin(abs(res4-resX))
#             prob2_1[kk] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#             resN3 = (Dr1-fit_val[0])/std_req1_aic[ii]
#             prob3_1[kk] = 1-norm.cdf(resN3)
#             resN4 = (Dr1-fit_val[0])/std_est[ii]
#             prob4_1[kk] = 1-norm.cdf(resN4)
#             tmp = dens_plt[ii]
#             prob5_1[kk] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
            
        

# for ii in np.arange(0,len(Dr_req),1):
#     Dr1 = Dr_req[ii]
#     prob1_1 = np.zeros(len(X_plot))
#     prob2_1 = np.zeros(len(X_plot))
#     prob3_1 = np.zeros(len(X_plot))
#     prob4_1 = np.zeros(len(X_plot))
#     prob5_1 = np.zeros(len(X_plot))
#     for jj in np.arange(0,len(X_plot),1):
#         
#         fit_valC = KRRC.fit(X_plot[jj])
#         resN_Cl = (Dr1-fit_valC[0])/stdC
#         prob1_1[jj] = 1-norm.cdf(resN_Cl)
#         fit_val = KRR.fit(X_plot[jj])
#         res4 = (Dr1-fit_val[0])
#         ind_req = np.argmin(abs(res4-resX))
#         prob2_1[jj] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
#         resN3 = (Dr1-fit_val[0])/std_req1_aic[jj]
#         prob3_1[jj] = 1-norm.cdf(resN3)
#         resN4 = (Dr1-fit_val[0])/std_est[jj]
#         prob4_1[jj] = 1-norm.cdf(resN4)
#         tmp = dens_plt[jj]
#         prob5_1[jj] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
#         
#     DemHaz1[ii] = np.sum(prob1_1 * np.rot90(Haz_diff))
#     DemHaz2[ii] = np.sum(prob2_1 * np.rot90(Haz_diff))
#     DemHaz3[ii] = np.sum(prob3_1 * np.rot90(Haz_diff))
#     DemHaz4[ii] = np.sum(prob4_1 * np.rot90(Haz_diff))
#     DemHaz5[ii] = np.sum(prob5_1 * np.rot90(Haz_diff))

# Haz_diff = np.abs(np.array([Haz_int[i + 1] - Haz_int[i] for i in range(len(Haz_int)-1)]))

## Loss of functionality single time

time_1 = 20
time_2 = 90
time_3 = 360
time_4 = 480

med_times = np.array([time_1,time_2,time_3,time_4])

std_rec = 0.75

time_req =  [180] #(np.linspace(10, 2000, 1)[:, None])

time_im1 = np.zeros(len(X_plot))
time_im2 = np.zeros(len(X_plot))
time_im3 = np.zeros(len(X_plot))
time_im4 = np.zeros(len(X_plot))
time_im5 = np.zeros(len(X_plot))

time_only = np.zeros((len(time_req),0))

for ii in np.arange(0,len(time_req),1):
    for kk in np.arange(0,len(X_plot),1):
        
        time_im1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P1_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P1_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P1_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P1_C[kk]
        
        
        time_im2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P2_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P2_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P2_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P2_C[kk]
        
        time_im3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P3_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P3_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P3_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P3_C[kk]
        
        time_im4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P4_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P4_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P4_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P4_C[kk]
        
        time_im5[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P5_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P5_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P5_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P5_C[kk]

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

# plt.plot(np.exp(X_plot),time_im1,color = (219/255,43/255,57/255,0.3),lineStyle = (0,(3,5,1,5,1,5)),linewidth=4.0,label = 'Case 1')
plt.scatter(np.exp(X_plot),time_im1,color = (219/255,43/255,57/255,0.3),s=50,label = 'Case 1',marker='o')
plt.plot(np.exp(X_plot),time_im2,color = (219/255,43/255,57/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
plt.plot(np.exp(X_plot),time_im3,color = (219/255,43/255,57/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
plt.plot(np.exp(X_plot),time_im4,color = (219/255,43/255,57/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
plt.plot(np.exp(X_plot),time_im5,color = (219/255,43/255,57/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')
        
plt.legend(frameon=False,loc='lower right')
plt.title('RC moment frame ($T^* = 180$ days)')
plt.xlabel('Sa(1.5s) [g]')
plt.ylabel('Exceedance probability')
plt.ylim((0, 1))
plt.show()


## Loss of functionality hazard

time_1 = 20
time_2 = 90
time_3 = 360
time_4 = 480

med_times = np.array([time_1,time_2,time_3,time_4])

std_rec = 0.75

time_req =  (np.linspace(10, 2000, 200)[:, None])

time_haz1 = np.zeros(len(time_req))
time_haz2 = np.zeros(len(time_req))
time_haz3 = np.zeros(len(time_req))
time_haz4 = np.zeros(len(time_req))
time_haz5 = np.zeros(len(time_req))

for ii in np.arange(0,len(time_req),1):
    
    time_im1 = np.zeros(len(X_plot))
    time_im2 = np.zeros(len(X_plot))
    time_im3 = np.zeros(len(X_plot))
    time_im4 = np.zeros(len(X_plot))
    time_im5 = np.zeros(len(X_plot))

    for kk in np.arange(0,len(X_plot),1):
        time_im1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P1_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P1_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P1_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P1_C[kk]
        
        
        time_im2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P2_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P2_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P2_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P2_C[kk]
        
        time_im3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P3_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P3_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P3_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P3_C[kk]
        
        time_im4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P4_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P4_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P4_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P4_C[kk]
        
        time_im5[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P5_S[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P5_M[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P5_Se[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P5_C[kk]
        
    time_haz1[ii] = np.sum(time_im1 * np.rot90(Haz_diff))
    time_haz2[ii] = np.sum(time_im2 * np.rot90(Haz_diff))
    time_haz3[ii] = np.sum(time_im3 * np.rot90(Haz_diff))
    time_haz4[ii] = np.sum(time_im4 * np.rot90(Haz_diff))
    time_haz5[ii] = np.sum(time_im5 * np.rot90(Haz_diff))
        
ax = plt.gca()

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

# plt.plot(time_req,time_haz1,color = (219/255,43/255,57/255,0.3),lineStyle = (0,(3,5,1,5,1,5)),linewidth=4.0,label = 'Case 1')
# # plt.plot(time_req,time_haz1,color = (80/255,114/255,60/255,0.4),lineStyle = 'dashed',linewidth=5.0,label = 'Case 1')
# plt.plot(time_req,time_haz2,color = (219/255,43/255,57/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
# plt.plot(time_req,time_haz3,color = (219/255,43/255,57/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
# plt.plot(time_req,time_haz4,color = (219/255,43/255,57/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
# plt.plot(time_req,time_haz5,color = (219/255,43/255,57/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')


plt.scatter(time_req,time_haz1,color = (41/255,51/255,92/255,0.3),label = 'Case 1',s=50,marker='o')# 
plt.plot(time_req,time_haz2,color = (41/255,51/255,92/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')# 
plt.plot(time_req,time_haz3,color = (41/255,51/255,92/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3') # 
plt.plot(time_req,time_haz4,color = (41/255,51/255,92/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4') # 
plt.plot(time_req,time_haz5,color = (41/255,51/255,92/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')


plt.legend(frameon=False,loc='lower left')
plt.title('Steel moment frame')
plt.xlabel('Time [days]')
plt.ylabel('Exceedance probability in 50 years')
# plt.ylim((0, 1))

ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

## Deaggregated risk

time_1 = 20
time_2 = 90
time_3 = 360
time_4 = 480

med_times = np.array([time_1,time_2,time_3,time_4])

std_rec = 0.75

time_req =  [2000]

time_haz1 = np.zeros(4)
time_haz2 = np.zeros(4)
time_haz3 = np.zeros(4)
time_haz4 = np.zeros(4)
time_haz5 = np.zeros(4)

for ii in np.arange(0,len(time_req),1):
    
    time_im1_d1 = np.zeros(len(X_plot))
    time_im1_d2 = np.zeros(len(X_plot))
    time_im1_d3 = np.zeros(len(X_plot))
    time_im1_d4 = np.zeros(len(X_plot))
    
    time_im2_d1 = np.zeros(len(X_plot))
    time_im2_d2 = np.zeros(len(X_plot))
    time_im2_d3 = np.zeros(len(X_plot))
    time_im2_d4 = np.zeros(len(X_plot))
    
    time_im3_d1 = np.zeros(len(X_plot))
    time_im3_d2 = np.zeros(len(X_plot))
    time_im3_d3 = np.zeros(len(X_plot))
    time_im3_d4 = np.zeros(len(X_plot))
    
    time_im4_d1 = np.zeros(len(X_plot))
    time_im4_d2 = np.zeros(len(X_plot))
    time_im4_d3 = np.zeros(len(X_plot))
    time_im4_d4 = np.zeros(len(X_plot))
    
    time_im5_d1 = np.zeros(len(X_plot))
    time_im5_d2 = np.zeros(len(X_plot))
    time_im5_d3 = np.zeros(len(X_plot))
    time_im5_d4 = np.zeros(len(X_plot))

    for kk in np.arange(0,len(X_plot),1):
        
        time_im1_d1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P1_S[kk] 
        time_im1_d2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P1_M[kk] 
        time_im1_d3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P1_Se[kk] 
        time_im1_d4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P1_C[kk]
        
        time_im2_d1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P2_S[kk] 
        time_im2_d2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P2_M[kk] 
        time_im2_d3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P2_Se[kk] 
        time_im2_d4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P2_C[kk]
        
        time_im3_d1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P3_S[kk] 
        time_im3_d2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P3_M[kk] 
        time_im3_d3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P3_Se[kk] 
        time_im3_d4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P3_C[kk]
        
        time_im4_d1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P4_S[kk] 
        time_im4_d2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P4_M[kk] 
        time_im4_d3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P4_Se[kk] 
        time_im4_d4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P4_C[kk]
        
        time_im5_d1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P5_S[kk] 
        time_im5_d2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P5_M[kk] 
        time_im5_d3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P5_Se[kk] 
        time_im5_d4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_4))/std_rec)) * P5_C[kk]
        
    time_haz1[0] = np.sum(time_im1_d1 * np.rot90(Haz_diff))
    time_haz1[1] = np.sum(time_im1_d2 * np.rot90(Haz_diff))
    time_haz1[2] = np.sum(time_im1_d3 * np.rot90(Haz_diff))
    time_haz1[3] = np.sum(time_im1_d4 * np.rot90(Haz_diff))
    
    time_haz2[0] = np.sum(time_im2_d1 * np.rot90(Haz_diff))
    time_haz2[1] = np.sum(time_im2_d2 * np.rot90(Haz_diff))
    time_haz2[2] = np.sum(time_im2_d3 * np.rot90(Haz_diff))
    time_haz2[3] = np.sum(time_im2_d4 * np.rot90(Haz_diff))
    
    time_haz3[0] = np.sum(time_im3_d1 * np.rot90(Haz_diff))
    time_haz3[1] = np.sum(time_im3_d2 * np.rot90(Haz_diff))
    time_haz3[2] = np.sum(time_im3_d3 * np.rot90(Haz_diff))
    time_haz3[3] = np.sum(time_im3_d4 * np.rot90(Haz_diff))
    
    time_haz4[0] = np.sum(time_im4_d1 * np.rot90(Haz_diff))
    time_haz4[1] = np.sum(time_im4_d2 * np.rot90(Haz_diff))
    time_haz4[2] = np.sum(time_im4_d3 * np.rot90(Haz_diff))
    time_haz4[3] = np.sum(time_im4_d4 * np.rot90(Haz_diff))
    
    time_haz5[0] = np.sum(time_im5_d1 * np.rot90(Haz_diff))
    time_haz5[1] = np.sum(time_im5_d2 * np.rot90(Haz_diff))
    time_haz5[2] = np.sum(time_im5_d3 * np.rot90(Haz_diff))
    time_haz5[3] = np.sum(time_im5_d4 * np.rot90(Haz_diff))
    
time_haz1 = time_haz1/np.sum(time_haz1)
time_haz2 = time_haz2/np.sum(time_haz2)
time_haz3 = time_haz3/np.sum(time_haz3)
time_haz4 = time_haz4/np.sum(time_haz4)
time_haz5 = time_haz5/np.sum(time_haz5)
        
ax = plt.gca()

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

# plt.plot(time_req,time_haz1,color = (219/255,43/255,57/255,0.3),lineStyle = (0,(3,5,1,5,1,5)),linewidth=4.0,label = 'Case 1')

# plt.plot(np.array([1,2,3,4]),time_haz1,color = (80/255,114/255,60/255,0.4),lineStyle = 'dashed', marker='s',linewidth=5.0, markersize=15,label = 'Case 1')
# # plt.plot(np.exp(X_plot),time_haz2,color = (80/255,114/255,60/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
# plt.plot(np.array([1,2,3,4]),time_haz2,color = (80/255,114/255,60/255,0.6),lineStyle = 'dotted', marker='s',linewidth=5.0, markersize=15,label = 'Case 2')
# plt.plot(np.array([1,2,3,4]),time_haz3,color = (80/255,114/255,60/255,0.8),lineStyle = 'dashdot', marker='s',linewidth=5.0, markersize=15,label = 'Case 3')
# plt.plot(np.array([1,2,3,4]),time_haz4,color = (80/255,114/255,60/255,0.8),lineStyle = 'solid', marker='s',linewidth=5.0, markersize=15,label = 'Case 4')


plt.scatter(np.array(['Slight','Moderate','Severe','Complete']),time_haz1,color = (41/255,51/255,92/255,0.3),label = 'Case 1',s=50,marker='o')# 
plt.plot(np.array(['Slight','Moderate','Severe','Complete']),time_haz2,color = (41/255,51/255,92/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')# 
plt.plot(np.array(['Slight','Moderate','Severe','Complete']),time_haz3,color = (41/255,51/255,92/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3') # 
plt.plot(np.array(['Slight','Moderate','Severe','Complete']),time_haz4,color = (41/255,51/255,92/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4') # 
plt.plot(np.array(['Slight','Moderate','Severe','Complete']),time_haz5,color = (41/255,51/255,92/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')


plt.legend(frameon=False,loc='upper left')
plt.title('Steel moment frame')
plt.xlabel('Damage state')
plt.ylabel('Probability mass')
# plt.ylim((0, 1))
# plt.xticks([1,2,3,4])

# ax.set_xscale('log')
# ax.set_yscale('log')
plt.show()

