
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


#set plotting defaults
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.grid'] = True

#name of the datafile
filename = 'test_data1.xls'
#Names of the x,y coordinate columns
x = 'x'
y = 'y'

data= pd.read_excel(filename)

#make a list of the variable names to pass to functions
variables = ['val']

#Generate a pandas dataframe of all pairs of data.  
#Set the function to return the unordered set of pairs
unordered=True
pairs = extract_pairs(data,x,y,variables,unordered)

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

#The exponential model parameters
sill = 15
rng = 10
nugget = 0
gram_type = 'SV'

#Add the parameters to the model
model = FIT_EXPONENTIAL(sill,rng,nugget,gram_type)

#Set the test target
target = [(7,14)]
#Run the ordinary kriging algorithm on the test points
OK = ORDINARY_KRIGING(model,data,x,y)
Z,EV,weights = OK.estimate(target)
#Convert the weights list to a matrix
weights = np.matrix(weights).T
#Convert to a dataframe
df = pd.DataFrame(np.round(weights,decimals=3),columns=target)
df['x'] = data[x]
df['y'] = data[y]
df.to_csv('weights_out.csv')

#Generate the meshgrid
xx,yy = make_meshgrid(data[x],data[y],h=0.2)
#Convert the grid points to a list of x,y tuples
grid_points = list(zip(xx.ravel(),yy.ravel()))

#Run the ordinary kriging algorithm
Z,EV,Weights = OK.estimate(grid_points)

#Determine the dimensions of the output figures
dx = np.max(xx)-np.min(xx)
dy = np.max(yy)-np.min(yy)
plot_x = 5
plot_y = plot_x*dy/dx

#Plot the predicted values
fig,ax = plt.subplots(figsize=(plot_x,plot_y))
#plt.title('Predicted Concentration')
plt.xlabel('X')
plt.ylabel('Y')
N=30
plot_contours(fig,ax,Z,xx,yy,N,cmap=plt.cm.viridis,alpha=0.8)
ax.scatter(data[x],data[y],c='y',s=80,edgecolors='k')
for i,txt in enumerate(data['val']):
    ax.annotate(str(txt),(data[x][i],data[y][i]))
plt.savefig('predicted.png')

#Plot the error variance
fig,ax = plt.subplots(figsize=(plot_x,plot_y))
#plt.title('Error Variance')
plt.xlabel('X')
plt.ylabel('Y')
N=30
plot_contours(fig,ax,EV,xx,yy,N,cmap=plt.cm.viridis,alpha=0.8)
ax.scatter(data[x],data[y],c='y',s=80,edgecolors='k')
for i,txt in enumerate(data['val']):
    ax.annotate(str(txt),(data[x][i],data[y][i]))
plt.savefig('error.png')


#The exponential model parameters
sill = 30000
rng = 10
nugget = 14
gram_type = 'SV'

#Run the ordinary kriging algorithm 
Z,EV,Weights2 = ordinary_kriging(model,data,x,y,grid_points)

#Check the difference in weights when the sill and nugget are changed
diff = np.matrix(Weights)-np.matrix(Weights2)
print('minimum weight difference: {} \nmaximum weight difference: {}'.format(diff.min(),diff.max()))
