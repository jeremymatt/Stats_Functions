# -*- coding: utf-8 -*-
#Author: Jeremy Matt
#Date: 9/16/19
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def cov(data,variables):
    """
    Calculates the covariance between two variables
    
    INPUTS
        data - pandas dataframe containing the data
        variables - list of the variable names
        
    OUTPUT
        covariance
    """
    #Calculate the means of each of the variables
    means = data[variables].mean()
    
    #Calculate the product of the differences from the means
    cov = (data[variables[0]]-means[variables[0]])*(data[variables[1]]-
          means[variables[1]])
    #Normalize by N
    cov = cov.mean()
    
    return cov

def std(data,variable):
    """
    calculates the standard deviation 
    """
    #Find the mean of the variable
    mean = data[variable].mean()
    #Square the differences from the mean
    std = (data[variable]-mean)**2
    #take the square root
    std = std.mean()**.5
    
    return std

def rho(data,variables):
    """
    calculates pearson's rho (correlation coefficient)
    
    INPUTS
        data - pandas dataframe containing the data
        variables - list of the variable names
        
    OUTPUT
        rho - Pearson's rho
    """
    #Calculate the covariance
    covariance = cov(data,variables)
    #Calculate the standard deviation of each variable
    std1 = std(data,variables[0])
    std2 = std(data,variables[1])
    
    #Normalize the covariance by the standard deviations
    rho = covariance/(std1*std2)
    return rho
    
def extract_pairs(data,x,y,variables):
    """
    Returns all pairs of points (forward and reverse) and the heads and tails 
    for each variable in the variables list
    
    INPUTS
        data - pandas dataframe containing the data
        x,y - the names of the columns containing the x and y coordinates
        variables - list of the variable names
        
    OUTPUT
        deltas - pandas dataframe containing all pairs of data points in both
                directions (i,j and j,i).  The data returned:
                    1. The dx and dy values
                    2. The euclidean distance
                    3. The angle in degrees
                    4. The parameter values for the head and tail for all 
                        variables specified
    """
    #Initialize an empty pandas dataframe
    deltas = pd.DataFrame()
    #For each point except the last, check all pairs
    #NOTE: This approach has a memory complexity of O(2*(N-1)!)
    for i in range(len(data)-1):
        #Init two empty temporary dataframes
        temp = pd.DataFrame()
        temp2 = pd.DataFrame()
        #Extract the current x and y position
        cur_x = data.loc[i,x]
        cur_y = data.loc[i,y]
        #Find the delta X and delta Y for each pair from the
        #current point to all other points
        temp['dx'] = data.loc[i+1:,x]-cur_x
        temp['dy'] = data.loc[i+1:,y]-cur_y
        #Find the delta X and delta Y for each pair from
        #all other points to the current point
        temp2['dx'] = cur_x-data.loc[i+1:,x]
        temp2['dy'] = cur_y-data.loc[i+1:,y]
        for var in variables:
            #Grab the head and tail for all vars for each pair from the
            #current point to all other points
            temp['head|'+var] = data.loc[i,var]
            temp['tail|'+var] = data.loc[i+1:,var]
            #Grab the head and tail for all vars for each pair from
            #all other points to the current point
            temp2['head|'+var] = data.loc[i+1:,var]
            temp2['tail|'+var] = data.loc[i,var]
        
        #Append the temp pairs to the deltas dataframe
        deltas = deltas.append(temp)
        deltas = deltas.append(temp2)
    
    #Reset the index
    deltas.reset_index(drop=True,inplace=True)
    #Insert a column for distance and calculate
    deltas.insert(2,'dist',value=np.nan)
    #calculate distance
    deltas['dist'] = (deltas['dx']**2+deltas['dy']**2)**.5
    #Insert a column for angle
    deltas.insert(2,'theta',value=np.nan)
    #Calculate theta
    deltas['theta'] = np.arctan2(deltas['dy'],deltas['dx'])*180/np.pi
    
    return deltas

def extract_lags(pairs,lag,theta,variable):
    """
    Extracts pairs that match the lag and angle and returns only the head 
    and tail for the specified variable
    
    INPUTS
        pairs - pandas dataframe containing all pairs of data points in both
                directions (i,j and j,i). The output of the extract_pairs() 
                function.  Includes the following
                    1. The dx and dy values
                    2. The euclidean distance
                    3. The angle in degrees
                    4. The parameter values for the head and tail for one or
                        more variables
        lag - the lag distance of interest
        theta - the direction of interest
        variable - the variable(s) for which to return the head and tail vals
                If two parameters are passed, the zero-index is the head and
                the one-index is the tail.
        
    OUTPUT
        lag_pairs - pandas dataframe containing only pairs that match the
                    specified lag and theta.  The data for the specified 
                    variable are placed in 'head' and 'tail' columns
    """
    
    #If only a single variable is passed, set both the head and tail variable
    #to the same value
    if len(variable)==1:
        variable = [variable,variable]
    
    #create a mask of pairs where the distance equals the lag
    m1 = pairs['dist']==lag
    #create a mask of pairs where the angle equals theta
    m2 = pairs['theta']==theta
    
    #Extract the data where both masks are true
    lag_pairs_all = pairs[m1&m2]
    #Extract the dx, dy, distance, and theta columns
    lag_pairs = pd.DataFrame(lag_pairs_all[['dx','dy','dist','theta']])
    #Extract the variable values for the head and tail
    lag_pairs['head'] = lag_pairs_all['head|'+variable[0]]
    lag_pairs['tail'] = lag_pairs_all['tail|'+variable[1]]
    
    return lag_pairs
  
def SV(data,variables):
    """
    Calculates and returns the semivariance between two variables
    
    INPUTS
        data - pandas dataframe containing the data
        variables - list of the variable names
        
    OUTPUT
        SV - the semivariance calculated from the data
    """
    #Find the number of input feature vectors
    N = data.shape[0]
    #Calculate the squares of the differences
    squares = (data[variables[0]]-data[variables[1]])**2
    #sum the squares
    sum_squares = squares.sum()
    #Normalize by twice the number of input feature vectors
    SV = sum_squares/(2*N)
    
    return SV

def plot_data(data,x,y,variable,cmap,display_fig=True):
    """
    Plots an x-y scatterplot of the data where color indicates the variable
    value at the point
    
    INPUTS
        data - pandas dataframe containing the data
        x,y - the names of the columns containing the x and y coordinates
        variable - The variable to plot
        display_fig - Display to screen only.  Provide filename to save to file
    """
    
    #Extract the variable values
    z = data[variable]
    
    #Initialize the figure
    fig,ax = plt.subplots(figsize=(10,4))
    
    #Create the color map
    normalize = mpl.colors.Normalize(vmin=min(z), vmax=max(z))
    colors = [cmap(normalize(value)) for value in z]
    
    #Plot the values, label the axes, add title, add color bar
    ax.scatter(data[x],data[y],color=colors,)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(variable)
    #Set the axes scales to be equal
    plt.axis('equal')
    #Add the colorbar
    cax, _ = mpl.colorbar.make_axes(ax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    
    if display_fig:
        fig.savefig(display_fig, bbox_inches="tight")
        plt.close()
        print('plotting to scatter plot of {} to file'.format(variable))