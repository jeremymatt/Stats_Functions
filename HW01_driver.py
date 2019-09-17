# -*- coding: utf-8 -*-
#Author: Jeremy Matt
#Date: 9/16/19

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


#Import user-defined functions
from geostats_functions import *

#start timer
tic = time.time()

#Flag to toggle display of figures to the console
plt_to_file_only = True

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

#Set the colormap
cmap = mpl.cm.get_cmap('viridis')
#Set plotting defaults
mpl.rcParams['font.size']=18
mpl.rcParams['axes.grid']=True
#Plot resistivity
plot_data(data,x,y,res,cmap,'res_scatterplot.png')
#Plot permeability
plot_data(data,x,y,perm,cmap,'perm_scatterplot.png')

#Problem 1
plt.plot(data[res],data[perm],'.')
plt.xlabel(res)
plt.ylabel(perm)
plt.savefig('perm_res_scatter.png', bbox_inches="tight")
#optionally close file
if plt_to_file_only:
    plt.close()

#make a list of the variable names to pass to functions
variables = [res,perm]

#Calculate the covariance
covariance = cov(data,variables)
print('Covariance: {:0.3f}'.format(covariance))
#Calculate the correlation coefficient
pearson_rho = rho(data,variables)
print('rho: {:0.3f}'.format(pearson_rho))

#Generate a pandas dataframe of all pairs of data.  Note: both (i,j) and (j,i)
#pairs are included
pairs = extract_pairs(data,x,y,variables)

#Define a list of the lags to be checked
lags = [3,9,24,36,54,72]
#lags = [3,9,24,36,54,72,96,168,225]
#Define the thetas to be checked
thetas = [0,90]
#Make a list of the names of the stats to be calculated
stat_names = ['rho','COV','SV']
#A list of the column headers for the head and tail in the lag pairs dataframe
col_headers = ['head','tail']
#parameter for which to generate hh plots
parameter = res

#Build a data structure to hold the results
#Init an empty dict
stats = {}
for theta in thetas:
    #init an empty dict for the current theta
    stats[theta] = {}
    #for each stat, init an empty list
    for stat in stat_names:
        stats[theta,stat] = []

for lag in lags:
    for theta in thetas:
        #Extract the pairs that match the current lag and theta
        lag_pairs = extract_lags(pairs,lag,theta,parameter)
        #If there are no pairs, set the statistics values to nan  and save in 
        #temporary variables
        if len(lag_pairs)==0:
            p_rho = np.nan
            COV = np.nan
            SVs = np.nan
        else:
            #Calculate the summary statistics and save in temporary variables
            p_rho = rho(lag_pairs,col_headers)
            COV = cov(lag_pairs,col_headers)
            SVs = SV(lag_pairs,col_headers)
            
            #Plot and format
            plt.figure()
            plt.plot(lag_pairs['head'],lag_pairs['tail'],'.')
            #Extract the max and min values from the lag pairs structure 
            #to use to plot the y=x line
            max_val = max([max(lag_pairs['head']),max(lag_pairs['tail'])])
            min_val = min([min(lag_pairs['head']),min(lag_pairs['tail'])])
            #Calculate an offset for the annotation
            offset = (max_val-min_val)/4
            #Extract the number of points N
            N = lag_pairs.shape[0]
            #add the annotation
            plt.title('theta:{}, lag:{}mm, N:{}'.format(theta,lag,N))
            plt.annotate('rho:{:0.3f}\nCOV:{:0.3f}\nSV:{:0.3f}'.format(p_rho,
                         COV,SVs),(min_val,max_val-offset))
            plt.xlabel('{}(h)'.format(parameter))
            plt.ylabel('{}\n(h+lag)'.format(parameter))
            #Plot the y=x line
            plt.plot([min_val,max_val],[min_val,max_val],'r--')
            plt.savefig('hh-plot_theta-{}_lag-{}'.format(round(theta),lag), 
                        bbox_inches="tight")
            #optionally close file
            if plt_to_file_only:
                plt.close()
        
        #Add the summary statistics to the data structure
        stats[theta,'rho'].append(p_rho)
        stats[theta,'COV'].append(COV)
        stats[theta,'SV'].append(SVs)

#For each stat and theta, generate and save a figure.   
for stat in stat_names:
    for theta in thetas:
        plt.figure()
        plt.plot(lags,stats[theta,stat],'*--')
        plt.title('theta:{}'.format(theta))
        plt.xlabel('lag')
        plt.ylabel(stat)
        plt.savefig('lag-vs-stat_theta-{}_stat-{}'.format(round(theta),stat), 
                    bbox_inches="tight")
        #optionally close file
        if plt_to_file_only:
            plt.close()
 
#End timer and print result       
toc = time.time()
print('Finished script in {:0.3f} sec'.format(toc-tic))