# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from geostats_functions import *


#load data
filename = 'Burea_subset.xls'
x = 'x (mm)'
y = 'y (mm)'
res = 'Water Resisitivity ohm-m'
perm = 'Permeability (mD)'
data= pd.read_excel(filename)

#data = data[:10]

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

stats = {}
stats[0] = {}
stats[0,'rho'] = []
stats[0,'COV'] = []
stats[0,'SV'] = []
stats[90] = {}
stats[90,'rho'] = []
stats[90,'COV'] = []
stats[90,'SV'] = []

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
            plt.figure()
            plt.plot(lag_pairs['head'],lag_pairs['tail'],'.')
            plt.title('theta:{}, lag:{}mm'.format(theta,lag))
            max_val = max([max(lag_pairs['head']),max(lag_pairs['tail'])])
            min_val = min([min(lag_pairs['head']),min(lag_pairs['tail'])])
            plt.plot([min_val,max_val],[min_val,max_val],'r--')
            p_rho = rho(lag_pairs,['head','tail'])
            COV = cov(lag_pairs,['head','tail'])
            SVs = SV(lag_pairs,['head','tail'])
            
        stats[theta,'rho'].append(p_rho)
        stats[theta,'COV'].append(COV)
        stats[theta,'SV'].append(SVs)
        
stat_names = ['rho','COV','SV']
for stat in stat_names:
    for theta in thetas:
        plt.figure()
        plt.plot(lags,stats[theta,stat],'*--')
        plt.title('theta:{}'.format(theta))
        plt.xlabel('lag')
        plt.ylabel(stat)