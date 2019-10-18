
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


#Flag to toggle display of figures to the console
plt_to_file_only = False

#set plotting defaults
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.grid'] = True

#name of the datafile
filename = 'test_data.xls'
#Names of the x,y coordinate columns
x = 'x'
y = 'y'

#load data into a pandas dataframe
#try: data
#except: 
#    data= pd.read_excel(filename)
#    print('loading data')
    
data= pd.read_excel(filename)

#make a list of the variable names to pass to functions
variables = ['val']

#Generate a pandas dataframe of all pairs of data.  Note: both (i,j) and (j,i)
#pairs are included
#Set the function to return the unordered set of pairs
unordered=True
try: pairs
except: pairs = extract_pairs(data,x,y,variables,unordered)

#Generate variable names to grab the correct columns from the pairs structure
variable = ['val','val']
sv_vars = ['head|{}'.format(variable[0]),'tail|{}'.format(variable[1])]

#Calculate the pair-by-pair semivariance and store in a pandas dataframe
data_sv = COV_by_pair(pairs,sv_vars)

#Extract just the distance and semivariance columns from  the dataframe
differences = pd.DataFrame(data_sv[['dist','COV']])
#Rename the dist and SV columns to generic names to work with the semivariogram
#functions
differences.rename(columns={'dist':'distance','COV':'delta'},inplace=True)
#
##Labels for the x and y axes of the plots
#labels = {}
#labels['xlabel'] = '$\Delta$'
#labels['ylabel'] = 'semivariance'
#
##Set the binning mode type:
#    #Bins of an equal distance width
#    #The number of bins equal to the square root of the number of points
#sv_mode = {'bins':'sqrt','mode':'eq.dist'}
#
##Call the semivariogram generation function
#sv_x,sv_y = generate_semivariogram(differences,labels,sv_mode)
#N = differences.shape[0]
#
##Store the binned semivariogram data in a tuple to
#sv_points = (sv_x,sv_y)
#
##Generate an array of x-points at which to fit the models
#x_vals = np.linspace(0,max(sv_x),100)
#
##The exponential model parameters
#sill = 15000
#rng = 10
#nugget = 0
#gram_type = 'COV'
##Fit the exponential model
#model_y = fit_exponential(x_vals,sill,rng,nugget,gram_type)
##Store the model x and y values
#model = (x_vals,model_y)
##Define the title and generate the plot
#title = 'exponential model fit\nN={}'.format(N)
#plot_model_fit(sv_points,model,labels,title)

target = (7,14)

dist = dist_to_target(data,x,y,target)
dist_mat = gen_dist_matrix(data,x,y)

#The exponential model parameters
sill = 15
rng = 10
nugget = 0
gram_type = 'SV'

model = FIT_EXPONENTIAL(sill,rng,nugget,gram_type)
cij = model.fit(dist_mat)
C_ij = fit_exponential(dist_mat,sill,rng,nugget,gram_type)
n_pts = C_ij.shape[0]
C_ij = np.row_stack((C_ij,np.ones(n_pts)))
C_ij = np.column_stack((C_ij,np.ones(n_pts+1)))
C_ij[-1,-1]=0
C_ij = np.matrix(C_ij)

xx,yy = make_meshgrid(data[x],data[y],h=0.2)

grid_points = list(zip(xx.ravel(),yy.ravel()))
Z = []
EV = []

for target in grid_points:
    target_dist = np.matrix(dist_to_target(data,x,y,target))
    C_io = fit_exponential(target_dist,sill,rng,nugget,gram_type)
    C_io = np.row_stack((C_io.T,[[1]]))
    W = np.linalg.solve(C_ij,C_io)
    prediction = np.matrix(data['val'])*W[:-1,:]
    Z.append(prediction[0,0])


fig,ax = plt.subplots(figsize=(8,8))
plot_contours(ax,Z,xx,yy,cmap=plt.cm.viridis,alpha=0.8)
ax.scatter(data[x],data[y],c='y',s=80,edgecolors='k')
for i,txt in enumerate(data['val']):
    plt.annotate(str(txt),(data[x][i],data[y][i]))
