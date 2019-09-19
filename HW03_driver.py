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
plt_to_file_only = False

#set plotting defaults
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.grid'] = True

#name of the datafile
filename = 'Burea_subset.xls'
#Names of the x,y coordinate columns
x = 'x (mm)'
y = 'y (mm)'
#names of the variable data columns
res = 'Water Resisitivity ohm-m'
perm = 'Permeability (mD)'
#load data into a pandas dataframe
try: data
except: 
    data= pd.read_excel(filename)
    print('loading data')

#make a list of the variable names to pass to functions
variables = [res,perm]

#Generate a pandas dataframe of all pairs of data.  Note: both (i,j) and (j,i)
#pairs are included
try: pairs
except: pairs = extract_pairs(data,x,y,variables)

#Define a list of the lags to be checked
lags = [3,6,9,12,15,18]
#lags = [3,9,24,36,54,72,96,168,225]
#Define the thetas to be checked
thetas = [0,90]
#Make a list of the names of the stats to be calculated
stat_names = ['rho','COV','SV']
#A list of the column headers for the head and tail in the lag pairs dataframe
col_headers = ['head','tail']
#parameter for which to generate hh plots
#res on the x-axis (near) and perm on the y-axis (far)
parameter = [res,perm]

#Build a data structure to hold the results
#Init an empty dict
stats = {}
for theta in thetas:
    #init an empty dict for the current theta
    stats[theta] = {}
    #for each stat, init an empty list
    for stat in stat_names:
        stats[theta,stat] = []
        
figs_per_col = 2
num_rows = int(np.ceil(len(lags)/figs_per_col))

for theta in thetas:
    f, ax = plt.subplots(nrows = num_rows,ncols = figs_per_col, sharex=True, sharey=True, figsize=(10, 12), dpi= 80, facecolor='w', edgecolor='k')
    for ind,lag in enumerate(lags):
        cur_row = ind//figs_per_col
        cur_col = ind%figs_per_col
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
            ax[cur_row,cur_col].plot(lag_pairs['head'],lag_pairs['tail'],'.')
            #Calculate an offset for the annotation
#            offset = (max_val-min_val)/4
            #Extract the number of points N
            N = lag_pairs.shape[0]
            #add the annotation
            ax[cur_row,cur_col].set_title('lag:{}mm, N:{}\nrho:{:0.3g}, COV:{:0.3g}, SV:{:0.3g}'.format(lag,N,p_rho,COV,SVs))
#            plt.annotate('rho:{:0.3f}\nCOV:{:0.3f}\nSV:{:0.3f}'.format(p_rho,COV,SVs),(min_val,max_val-offset))
            if cur_row == num_rows-1:
                ax[cur_row,cur_col].set_xlabel('{}(h)'.format(parameter[0]))
            if cur_col == 0:
                ax[cur_row,cur_col].set_ylabel('{}\n(h+lag)'.format(parameter[1]))
            
            #Add the summary statistics to the data structure
            stats[theta,'rho'].append(p_rho)
            stats[theta,'COV'].append(COV)
            stats[theta,'SV'].append(SVs)
    f.suptitle('Theta: {}'.format(theta))
    f.savefig('hh-plots_theta-{}'.format(round(theta)), 
                bbox_inches="tight")
    #optionally close file
    if plt_to_file_only:
        plt.close()
        

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