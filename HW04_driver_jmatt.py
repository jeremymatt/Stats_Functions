
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
try: data
except: 
    data= pd.read_excel(filename)
    print('loading data')

#make a list of the variable names to pass to functions
variables = ['val']

#Generate a pandas dataframe of all pairs of data.  Note: both (i,j) and (j,i)
#pairs are included
#Set the function to return the unordered set of pairs
unordered=True
try: pairs
except: pairs = extract_pairs(data,x,y,variables,unordered)

#Generate variable names to grab the correct columns from the pairs structure
variable = [perm,perm]
sv_vars = ['head|{}'.format(variable[0]),'tail|{}'.format(variable[1])]

#Calculate the pair-by-pair semivariance and store in a pandas dataframe
data_sv = SV_by_pair(pairs,sv_vars)

#Extract just the distance and semivariance columns from  the dataframe
differences = pd.DataFrame(data_sv[['dist','SV']])
#Rename the dist and SV columns to generic names to work with the semivariogram
#functions
differences.rename(columns={'dist':'distance','SV':'delta'},inplace=True)

#Labels for the x and y axes of the plots
labels = {}
labels['xlabel'] = '$\Delta$ distance (mm)'
labels['ylabel'] = 'semivariance'

#Set the binning mode type:
    #Bins of an equal distance width
    #The number of bins equal to the square root of the number of points
sv_mode = {'bins':'sqrt','mode':'eq.dist'}

#Call the semivariogram generation function
x,y = generate_semivariogram(differences,labels,sv_mode)
N = differences.shape[0]

#Store the binned semivariogram data in a tuple to
sv_points = (x,y)

#Generate an array of x-points at which to fit the models
x_vals = np.linspace(0,max(x),100)

#The exponential model parameters
sill = 6500
rng = 30
nugget = 1000
#Fit the exponential model
model_y = fit_exponential(x_vals,sill,rng,nugget)
#Store the model x and y values
model = (x_vals,model_y)
#Define the title and generate the plot
title = 'exponential model fit\nN={}'.format(N)
plot_model_fit(sv_points,model,labels,title)



#The gaussian model parameters
sill = 6500
rng = 20
nugget = 1000
model_y = fit_gaussian(x_vals,sill,rng,nugget)
#Store the model x and y values
model = (x_vals,model_y)
#Define the title and generate the plot
title = 'gaussian model fit\nN={}'.format(N)
plot_model_fit(sv_points,model,labels,title)

#The spherical model parameters
sill = 6500
rng = 30
nugget = 1000
#Fit the spherical model
model_y = fit_spherical(x_vals,sill,rng,nugget)
#Store the model x and y values
model = (x_vals,model_y)
#Define the title and generate the plot
title = 'spherical model fit\nN={}'.format(N)
plot_model_fit(sv_points,model,labels,title)
