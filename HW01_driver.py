# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from geostats_functions import *

#name of the datafile
filename = 'Burea_subset.xls'
#Names of the x,y coordinate columns
x = 'x (mm)'
y = 'y (mm)'
#names of the variable data columns
res = 'Water Resisitivity ohm-m'
perm = 'Permeability (mD)'
#load data into a pandas dataframe
data= pd.read_excel(filename)

#data = data[:10]

###############
###############
#PLOT THE DATA
###############
###############

x_diffs = np.diff(data[x])
y_diffs = np.diff(data[y])
plt.figure()
plt.plot(data.iloc[:-1][x],x_diffs,'.')
plt.xlabel('x (mm)')
plt.ylabel('dx (mm)')
plt.figure()
plt.plot(data.iloc[:-1][y],y_diffs,'.')

plt.plot(data[x],data[y],'.')


#Problem 1
plt.plot(data[res],data[perm],'.')
plt.xlabel(res)
plt.ylabel(perm)

variables = [res,perm]

covariance = cov(data,variables)
pearson_rho = rho(data,variables)

pairs = extract_pairs(data,x,y,variables)


lags = [3,9,24,36,54,72]
thetas = [0,90]
stat_names = ['rho','COV','SV']

#Build a data structure to hold the results
#Init an empty dict
stats = {}
for theta in thetas:
    #for each theta, init an empty dict 
    stats[theta] = {}
    for stat in stat_names:
        stats[theta,stat] = []

for lag in lags:
    for theta in thetas:
        lag_pairs = extract_lags(pairs,lag,theta,res)
        if len(lag_pairs)==0:
            p_rho = np.nan
            COV = np.nan
            SV = np.nan
        else:
            #
            #
            #ADD N VALUES
            #ALSO DISCUSS THE REASON WHY MORE IN X than Y
            #SAID IN CLASS THAT THE SLAB WAS INTENTIONALLY SET ON THE
            #MACHINE THA WAY
            #
            #
            p_rho = rho(lag_pairs,['head','tail'])
            COV = cov(lag_pairs,['head','tail'])
            SVs = SV(lag_pairs,['head','tail'])
            
            
            plt.figure()
            plt.plot(lag_pairs['head'],lag_pairs['tail'],'.')
            max_val = max([max(lag_pairs['head']),max(lag_pairs['tail'])])
            min_val = min([min(lag_pairs['head']),min(lag_pairs['tail'])])
            rng = max_val-min_val
            plt.annotate('theta:{}\nlag:{}mm\nrho:{:0.3f}\nCOV:{:0.3f}\nSV:{:0.3f}'.format(theta,lag,p_rho,COV,SVs),(min_val,max_val-rng/4))
            plt.plot([min_val,max_val],[min_val,max_val],'r--')
            
        stats[theta,'rho'].append(p_rho)
        stats[theta,'COV'].append(COV)
        stats[theta,'SV'].append(SVs)
        
for stat in stat_names:
    for theta in thetas:
        plt.figure()
        plt.plot(lags,stats[theta,stat],'*--')
        plt.title('theta:{}'.format(theta))
        plt.xlabel('lag')
        plt.ylabel(stat)