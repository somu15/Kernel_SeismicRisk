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

## Data

wb = xlrd.open_workbook('SMF_PD_SF1.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM = []
PD = []
for ii in np.arange(1,372,1):
    if sheet.cell_value(ii,0)>0 and sheet.cell_value(ii,0)<10 and sheet.cell_value(ii,1)<0.1 and sheet.cell_value(ii,1)>0:
        count = count + 1
        IM.append(sheet.cell_value(ii,0))
        PD.append(sheet.cell_value(ii,1))
        
lnIM = np.array(np.log(IM))
lnPD = np.array(np.log(PD))    

X_plot = np.log(np.linspace(0.05, 2, 100)[:, None])
X_plot1 = np.log(np.linspace(0.05, 2.04, 101)[:, None])

## StatsModel Fit
KRR = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'll',bw='cv_ls') # np.array([1e20])
KRR_fit = KRR.fit(X_plot)
KRR_fit_res = KRR.fit(lnIM)

KRR1 = statsmodels.nonparametric.kernel_regression.KernelReg(lnPD, lnIM, 'c',reg_type = 'lc',bw='cv_ls') # np.array([1e20])
KRR_fit1 = KRR1.fit(X_plot)
KRR_fit_res1 = KRR1.fit(lnIM)
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


## Scikit kernel


clf = KernelRidge(alpha=0.0)
clf.fit(lnIM.reshape(-1, 1), lnPD.reshape(-1, 1)) 

sck_pred = clf.predict(X_plot)

ax = plt.gca()
ax.scatter(IM,PD,color = (41/255,51/255,92/255,0.5),s=200,marker="d")
plt.plot(np.exp(X_plot),np.exp(sck_pred))
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


resX_plt = np.empty((100,0))
dens_plt = np.empty((100,0))

N_samps = 40
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
    
    
    # res_fit = statsmodels.nonparametric.kernel_regression.KernelReg(res_req, lnIM, 'c',reg_type = 'lc',bw='cv_ls')
    
    # kde = KernelDensity(kernel='gaussian',bandwidth=np.power(len(res_knn[KK1[1]]),-1/5)*1.06*np.std(res_knn[KK1[1]])).fit(res_knn[KK1[1]].reshape(-1, 1))
    kde = KernelDensity(kernel='gaussian',bandwidth=np.power(N_samps,-1/5)*1.06*np.std(res_knn[KK1[1]])).fit(res_knn[KK1[1]].reshape(-1, 1))

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
yint = np.arange(0.05,2.0,0.067242)
lc = multiline(resX_plt, dens_plt, yint, cmap='YlGn', lw=4)

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)
axcb = fig.colorbar(lc)
axcb.set_label('Sa(0.53s) [g]')
ax.set_title('Wood shear wall')
plt.xlabel('Residual')
plt.ylabel('Density')

kde1 = KernelDensity(kernel='gaussian',bandwidth=np.power(len(IM),-1/5)*1.06*np.std(res_knn)).fit(res_knn.reshape(-1, 1))
log_dens1 = kde1.score_samples(resX)
plt.plot(resX,np.exp(log_dens1),color='k',label='All samples',linewidth=5.0,lineStyle = '--')
plt.legend(frameon=False,loc='upper left')
plt.show()

# WOOD: YlGn
# SMF: winter
# RCMF: YlOrRd


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
ax.scatter(IM,PD,color = (41/255,51/255,92/255,0.5),s=200,marker="d")
ax.plot(np.exp(X_plot),np.exp(KRRC.fit(X_plot)[0]),color = 'k',linewidth=3.0,lineStyle = '--')
ax.plot(np.exp(X_plot),np.exp(KRR_fit[0]),color = 'k',linewidth=3.0,lineStyle = '-')
ax.plot(np.exp(X_plot),np.exp(KRR_fit1[0]),color = 'k',linewidth=3.0,lineStyle = '-.')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Sa(1.33s) [g]')
plt.ylabel('Peak Interstory Drift')
plt.title('Steel moment frame')
# plt.legend(frameon=False)
plt.show()

## Error squared
res_req = (lnPD-KRR_fit_res[0])**2
res_req1 = (lnPD-KRR_fit_res1[0])**2

res_fit = statsmodels.nonparametric.kernel_regression.KernelReg(res_req, lnIM, 'c',reg_type = 'll',bw='cv_ls')
res_fit1 = statsmodels.nonparametric.kernel_regression.KernelReg(res_req, lnIM, 'c',reg_type = 'lc',bw='cv_ls')

res_pred = res_fit.fit(X_plot)
res_pred1 = res_fit1.fit(X_plot)

std_req = sqrt(res_pred[0])
std_req1 = sqrt(res_pred1[0])

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}
figure(num=None, figsize=(8, 7), facecolor='w', edgecolor='k')
plt.rc('font', **font)
ax = plt.gca()

plt.scatter(np.exp(lnIM),res_req,color = (219/255,43/255,57/255,0.5),s=200,marker="o")

plt.plot(np.exp(X_plot),res_pred[0],color = 'k',linewidth=3.0,lineStyle = '-',label='Local linear')

plt.plot(np.exp(X_plot),res_pred1[0],color = 'k',linewidth=3.0,lineStyle = '-.',label='Local constant')

plt.legend()
plt.xlabel('Sa(1.5s) [g]')
plt.ylabel('Error squared')
plt.title('RC moment frame')
plt.legend(frameon=False)
ax = plt.gca()
ax.set_xscale('log')
plt.show()


## Standard deviation comparison

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

plt.plot(np.array([np.exp(X_plot[0]), np.exp(X_plot[99])]), np.array([stdC,stdC]),color = (80/255,114/255,60/255),lineStyle = 'solid',linewidth=5.0,label = 'Linear regression')
plt.plot(np.exp(X_plot),std_req,color = (80/255,114/255,60/255),lineStyle = 'dashed',linewidth=5.0,label = 'Local linear')
plt.plot(np.exp(X_plot),std_req1,color = (80/255,114/255,60/255),lineStyle = 'dotted',linewidth=5.0,label = 'Local constant')
plt.plot(np.exp(X_plot),std_est,color = (80/255,114/255,60/255),lineStyle = 'dashdot',linewidth=5.0,label = 'K nearest neighbor')

ax = plt.gca()
ax.set_xscale('log')
plt.legend(frameon=False,loc='lower left')
plt.title('Wood shear wall')
plt.xlabel('Sa(0.53s) [g]')
plt.ylabel('Standard deviation')
# plt.yticks([0.4, 0.5, 0.6, 0.7])
plt.show()

## Compute fragility

Dr = np.log(0.05)

prob1 = np.zeros(len(X_plot))
prob2 = np.zeros(len(X_plot))
prob3 = np.zeros(len(X_plot))
prob4 = np.zeros(len(X_plot))
prob5 = np.zeros(len(X_plot))

pdf1 = np.zeros(len(X_plot))
pdf2 = np.zeros(len(X_plot))


for ii in np.arange(0,len(X_plot),1):
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr-fit_valC[0])/stdC
    prob1[ii] = 1-norm.cdf(resN_Cl)
    
    pdf1[ii] = norm.pdf(resN_Cl)

    
    
    fit_val = KRR.fit(X_plot[ii])
    resN2 = (Dr-fit_val[0])/std_req[ii]
    resN3 = (Dr-fit_val[0])/std_req1[ii]
    prob2[ii] = 1-norm.cdf(resN2)
    prob3[ii] = 1-norm.cdf(resN3)
    
    res4 = (Dr-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob4[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    pdf2[ii] = np.exp(log_dens1[ind_req])
    tmp = dens_plt[ii]
    prob5[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    
    # samps_rec[ind] = N_samps
    # std_est[ind] = np.std(res_knn[KK1[1]])
    
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

# plt.plot(np.exp(X_plot),prob1,color = (219/255,43/255,57/255,0.3),linewidth=4.0,label = 'Case 1',marker='o')
plt.scatter(np.exp(X_plot),prob1,color = (80/255,114/255,60/255,0.3),s=50,marker='o',label='Case 1')
plt.plot(np.exp(X_plot),prob2,color = (80/255,114/255,60/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
plt.plot(np.exp(X_plot),prob3,color = (80/255,114/255,60/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
plt.plot(np.exp(X_plot),prob4,color = (80/255,114/255,60/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
plt.plot(np.exp(X_plot),prob5,color = (80/255,114/255,60/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')

# plt.legend(frameon=False,loc='lower right')
plt.title('Wood shear wall (CP)')
plt.xlabel('Sa(0.53s) [g]')
plt.ylabel('Exceedance probability')
plt.ylim((0, 1))
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()
    

## Compute risk

Dr1 = np.log(0.007)
Dr2 = np.log(0.025)
Dr3 = np.log(0.05)

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

pdf1 = np.zeros(len(X_plot))
pdf2 = np.zeros(len(X_plot))


for ii in np.arange(0,len(X_plot),1):
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr1-fit_valC[0])/stdC
    prob1_1[ii] = 1-norm.cdf(resN_Cl)
    pdf1[ii] = norm.pdf(resN_Cl)
    fit_val = KRR.fit(X_plot[ii])
    resN2 = (Dr1-fit_val[0])/std_req[ii]
    resN3 = (Dr1-fit_val[0])/std_req1[ii]
    prob2_1[ii] = 1-norm.cdf(resN2)
    prob3_1[ii] = 1-norm.cdf(resN3)
    res4 = (Dr1-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob4_1[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    pdf2[ii] = np.exp(log_dens1[ind_req])
    tmp = dens_plt[ii]
    prob5_1[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr2-fit_valC[0])/stdC
    prob1_2[ii] = 1-norm.cdf(resN_Cl)
    pdf1[ii] = norm.pdf(resN_Cl)
    fit_val = KRR.fit(X_plot[ii])
    resN2 = (Dr2-fit_val[0])/std_req[ii]
    resN3 = (Dr2-fit_val[0])/std_req1[ii]
    prob2_2[ii] = 1-norm.cdf(resN2)
    prob3_2[ii] = 1-norm.cdf(resN3)
    res4 = (Dr2-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob4_2[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    pdf2[ii] = np.exp(log_dens1[ind_req])
    tmp = dens_plt[ii]
    prob5_2[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))
    
    fit_valC = KRRC.fit(X_plot[ii])
    resN_Cl = (Dr3-fit_valC[0])/stdC
    prob1_3[ii] = 1-norm.cdf(resN_Cl)
    pdf1[ii] = norm.pdf(resN_Cl)
    fit_val = KRR.fit(X_plot[ii])
    resN2 = (Dr3-fit_val[0])/std_req[ii]
    resN3 = (Dr3-fit_val[0])/std_req1[ii]
    prob2_3[ii] = 1-norm.cdf(resN2)
    prob3_3[ii] = 1-norm.cdf(resN3)
    res4 = (Dr3-fit_val[0])
    ind_req = np.argmin(abs(res4-resX))
    prob4_3[ii] = 1-trapz(np.exp(log_dens1[0:ind_req+1:1]),np.rot90(resX[0:ind_req+1:1]))
    pdf2[ii] = np.exp(log_dens1[ind_req])
    tmp = dens_plt[ii]
    prob5_3[ii] = 1-trapz(tmp[0:ind_req+1:1],np.rot90(resX[0:ind_req+1:1]))


P1_oo = 1-prob1_1
P1_io = prob1_1-prob1_2
P1_ls = prob1_2-prob1_3
P1_cp = prob1_3

P2_oo = 1-prob2_1
P2_io = prob2_1-prob2_2
P2_ls = prob2_2-prob2_3
P2_cp = prob2_3

P3_oo = 1-prob3_1
P3_io = prob3_1-prob3_2
P3_ls = prob3_2-prob3_3
P3_cp = prob3_3

P4_oo = 1-prob4_1
P4_io = prob4_1-prob4_2
P4_ls = prob4_2-prob4_3
P4_cp = prob4_3

P5_oo = 1-prob5_1
P5_io = prob5_1-prob5_2
P5_ls = prob5_2-prob5_3
P5_cp = prob5_3

## Seismic hazard

wb = xlrd.open_workbook('Sa133_LA.xlsx') 
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

## Loss of functionality single time

time_1 = 20
time_2 = 90
time_3 = 360

med_times = np.array([time_1,time_2,time_3])

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
        
        time_im1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P1_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P1_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P1_cp[kk]
        
        
        time_im2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P2_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P2_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P2_cp[kk]
        
        time_im3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P3_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P3_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P3_cp[kk]
        
        time_im4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P4_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P4_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P4_cp[kk]
        
        time_im5[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P5_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P5_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P5_cp[kk]

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)

# plt.plot(np.exp(X_plot),time_im1,color = (219/255,43/255,57/255,0.3),lineStyle = (0,(3,5,1,5,1,5)),linewidth=4.0,label = 'Case 1')
plt.scatter(np.exp(X_plot),time_im1,color = (80/255,114/255,60/255,0.3),s=50,label = 'Case 1',marker='o')
plt.plot(np.exp(X_plot),time_im2,color = (80/255,114/255,60/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
plt.plot(np.exp(X_plot),time_im3,color = (80/255,114/255,60/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
plt.plot(np.exp(X_plot),time_im4,color = (80/255,114/255,60/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
plt.plot(np.exp(X_plot),time_im5,color = (80/255,114/255,60/2555,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')
        
# plt.legend(frameon=False,loc='lower right')
plt.title('Wood shear wall ($T^* = 180$ days)')
plt.xlabel('Sa(0.53s) [g]')
plt.ylabel('Exceedance probability')
plt.ylim((0, 1))
plt.show()


## Loss of functionality hazard

time_1 = 20
time_2 = 90
time_3 = 360

med_times = np.array([time_1,time_2,time_3])

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
        time_im1[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P1_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P1_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P1_cp[kk]
        
        
        time_im2[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P2_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P2_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P2_cp[kk]
        
        time_im3[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P3_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P3_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P3_cp[kk]
        
        time_im4[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P4_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P4_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P4_cp[kk]
        
        time_im5[kk] = (1-norm.cdf((np.log(time_req[ii])-np.log(time_1))/std_rec)) * P5_io[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_2))/std_rec)) * P5_ls[kk] + (1-norm.cdf((np.log(time_req[ii])-np.log(time_3))/std_rec)) * P5_cp[kk]
        
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
plt.scatter(time_req,time_haz1,color = (80/255,114/255,60/255,0.3),s=50,marker='o',label = 'Case 1')
plt.plot(time_req,time_haz2,color = (80/255,114/255,60/255,0.5),lineStyle = 'dashed',linewidth=5.0,label = 'Case 2')
plt.plot(time_req,time_haz3,color = (80/255,114/255,60/255,0.7),lineStyle = 'dotted',linewidth=5.0,label = 'Case 3')
plt.plot(time_req,time_haz4,color = (80/255,114/255,60/255,0.85),lineStyle = 'dashdot',linewidth=5.0,label = 'Case 4')
plt.plot(time_req,time_haz5,color = (80/255,114/255,60/255,1),lineStyle = 'solid',linewidth=5.0,label = 'Case 5')
        
# plt.legend(frameon=False,loc='lower left')
plt.title('Wood shear wall')
plt.xlabel('Time [days]')
plt.ylabel('Exceedance probability in 50 years')
# plt.ylim((0, 1))

ax.set_xscale('log')
ax.set_yscale('log')
plt.show()



