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



mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2.5

## Responses scatter

wb = xlrd.open_workbook('RC_PD_SF1_C.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM1 = []
PD1 = []
for ii in np.arange(1,382,1):
    if sheet.cell_value(ii,0)>0 and sheet.cell_value(ii,0)<10 and sheet.cell_value(ii,1)<0.1 and sheet.cell_value(ii,1)>0:
        count = count + 1
        IM1.append(sheet.cell_value(ii,0))
        PD1.append(sheet.cell_value(ii,1))
        
wb = xlrd.open_workbook('SMF_PD_SF1.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM2 = []
PD2 = []
for ii in np.arange(1,382,1):
    if sheet.cell_value(ii,0)>0 and sheet.cell_value(ii,0)<10 and sheet.cell_value(ii,1)<0.1 and sheet.cell_value(ii,1)>0:
        count = count + 1
        IM2.append(sheet.cell_value(ii,0))
        PD2.append(sheet.cell_value(ii,1))
        
wb = xlrd.open_workbook('WOOD_PD_SF1_C.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM3 = []
PD3 = []
for ii in np.arange(1,382,1):
    if sheet.cell_value(ii,0)>0 and sheet.cell_value(ii,0)<10 and sheet.cell_value(ii,1)<0.1 and sheet.cell_value(ii,1)>0:
        count = count + 1
        IM3.append(sheet.cell_value(ii,0))
        PD3.append(sheet.cell_value(ii,1))

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}
figure(num=None, figsize=(10, 7), facecolor='w', edgecolor='k')
plt.rc('font', **font)
ax = plt.gca()
ax.scatter(IM1,PD1,color = (219/255,43/255,57/255,0.5),s=200,label='RC moment frame',marker="o")
ax.scatter(IM2,PD2,color = (41/255,51/255,92/255,0.5),s=200,label='Steel moment frame',marker="d")
ax.scatter(IM3,PD3,color = (80/255,114/255,60/255,0.5),s=200,label='Wood shear wall',marker="s")
# ax.set_yscale('log')
# ax.set_xscale('log')
plt.xlabel('Intensity Measure')
plt.ylabel('Peak Interstory Drift')
plt.legend(frameon=False,loc='upper left')
plt.show()

## Hazard curves

X_plot = np.log(np.linspace(0.05, 4.0, 100)[:, None])
X_plot1 = np.log(np.linspace(0.05, 4.04, 101)[:, None])

wb = xlrd.open_workbook('Sa15_LA.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM_haz1 = []
Haz1 = []
for ii in np.arange(1,51,1):
    count = count + 1
    IM_haz1.append(sheet.cell_value(ii,0))
    Haz1.append(sheet.cell_value(ii,1))

int_fun = interp1d(np.log(IM_haz1),np.log(Haz1))
Haz_int1 = np.exp(int_fun(X_plot1))

Haz_diff1 = np.abs(np.array([Haz_int1[i + 1] - Haz_int1[i] for i in range(len(Haz_int1)-1)]))

wb = xlrd.open_workbook('Sa133_LA.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM_haz2 = []
Haz2 = []
for ii in np.arange(1,51,1):
    count = count + 1
    IM_haz2.append(sheet.cell_value(ii,0))
    Haz2.append(sheet.cell_value(ii,1))

int_fun = interp1d(np.log(IM_haz2),np.log(Haz2))
Haz_int2 = np.exp(int_fun(X_plot1))

Haz_diff2 = np.abs(np.array([Haz_int2[i + 1] - Haz_int2[i] for i in range(len(Haz_int2)-1)]))

wb = xlrd.open_workbook('Sa053_LA.xlsx') 
sheet = wb.sheet_by_index(0)
# sheet.cell_value(1, 0)
count = 0
IM_haz3 = []
Haz3 = []
for ii in np.arange(1,51,1):
    count = count + 1
    IM_haz3.append(sheet.cell_value(ii,0))
    Haz3.append(sheet.cell_value(ii,1))

int_fun = interp1d(np.log(IM_haz3),np.log(Haz3))
Haz_int3 = np.exp(int_fun(X_plot1))

Haz_diff3 = np.abs(np.array([Haz_int3[i + 1] - Haz_int3[i] for i in range(len(Haz_int3)-1)]))

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}
# figure(num=None, figsize=(10, 7), facecolor='w', edgecolor='k')
plt.rc('font', **font)
ax = plt.gca()
plt.plot(IM_haz1, Haz1,color = (219/255,43/255,57/255),lineStyle = 'solid',linewidth=4.0,label = 'T = 1.5s')
plt.plot(IM_haz2, Haz2,color = (41/255,51/255,92/255),lineStyle = 'solid',linewidth=4.0,label = 'T = 1.33s')
plt.plot(IM_haz3, Haz3,color = (80/255,114/255,60/255),lineStyle = 'solid',linewidth=4.0,label = 'T = 0.53s')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Intensity Measure')
plt.ylabel('Exceedance probability in 50 years')
plt.legend(frameon=False,loc='lower left')
plt.xlim((0.01, 4))
plt.title('Seismic hazard curves for Los Angeles, CA')
plt.show()