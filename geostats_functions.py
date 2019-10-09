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
    
def extract_pairs(data,x,y,variables,unordered=False):
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
            if not unordered:
                temp2['head|'+var] = data.loc[i+1:,var]
                temp2['tail|'+var] = data.loc[i,var]
        
        #Append the temp pairs to the deltas dataframe
        deltas = deltas.append(temp)
        if not unordered:
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


  
def SV_by_pair(data,variables):
    """
    Calculates and returns the pair-by-pair semivariance between two variables
    
    INPUTS
        data - pandas dataframe containing the data
        variables - list of the variable names
        
    OUTPUT
        SV - the semivariance calculated from the data
    """
    #Calculate the squares of the differences
    squares = (data[variables[0]]-data[variables[1]])**2
    #Divide the squares by 2
    SV = squares/(2)
    
    data['SV'] = SV
    
    return data


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
        
def plot_semivariogram(differences,labels,sv_mode):
    """
    Function to plot the semivariogram
    
    INPUTS: 
        differences - A pandas dataframe containing a column of distances and
                    'differences' (correlation, semivariance, or covariance)
        labels - The axis labels for the plot
                    labels[0] ==> xlabel
                    labels[1] ==> ylabel
        sv_mode - The binning sv_mode:
                    --> sv_mode={'bins':N ,'mode':'eq.dist'}
                        fixed number of N bins equally spaced between min
                        and max distance.  The default is to use the sqrt 
                        of the number of points
                    --> sv_mode={'bins':N ,'mode':'eq.pts'}
                        fixed number of N bins each with the same number of
                        points
                    --> sv_mode = {'bins':[0,2,4,6,8,10,50,100,180],'mode':'user.div'}
                        strictly increasing list of bin divisions.  
                        Calculations will not be performed
                        outside {min(sv_mode['bins']):max(sv_mode['bins'])}
                    --> sv_mode['bins'] = 'sqrt' 
                        Uses the square root of the number of data points as
                        the number of bins
                    
                    NOTE: User bin divisions are in units of the distances
                    matrix
    
    OUTPUTS
        fig_vars - the handles of the 
        x - The average distances of the points in each bin
        y - The average difference of the points in each bin
        bins - the bin divisions
"""
    #Find the number of data points
    num_values = differences.shape[0]
    #If requested by the user, set the number of bins equal to the square
    #root of the number of data points
    if sv_mode['bins'] == 'sqrt':
        sv_mode['bins'] = np.round(num_values**0.5).astype(int)
       
    #CHECK TO SEE IF THE DISTANCES ARE SORTED FROM SMALLEST TO LARGEST
    #Calculate the difference in distance between point(i) and point(i+1) for
    #i in [0,N-1]
    diffs = np.diff(differences['distance'])
    #Find the minimum change in distance
    min_diff = min(diffs)
    #If the distance decreases at any point, sort the 
    if min_diff<0:
        print('sorting the differences')
        differences.sort_values(by='distance',inplace=True)
        differences.reset_index(inplace=True,drop=True)
        
    
    #Find the indices of the bin divisions depending on the mode
    #Divide the semivariogram into bins of equal width distance
    if sv_mode['mode'] == 'eq.dist':
        #Find the min and max distances
        minDist = np.floor(differences['distance'].min())
        maxDist = np.ceil(differences['distance'].max())
        #Generate the bin divisions in terms of distances
        bin_divs = np.linspace(minDist,maxDist,sv_mode['bins']+1)
        #Convert distances to indices
        bin_inds = bin_divs2inds(differences,bin_divs,'distance',remove_empty=True)
    #Divide the semivariogram into bins each with an equal number of points
    elif sv_mode['mode'] == 'eq.pts':
        #Break the points up into bins each containing approximately equal
        #numbers of points
        bin_inds = np.round(np.linspace(0,differences.shape[0],sv_mode['bins']+1)).astype(int)
    elif sv_mode['mode'] == 'user.div':
        #Convert the user-provided distance bins into indices
        bin_inds = bin_divs2inds(differences,np.array(sv_mode['bins']),'distance',remove_empty=False)
    else:
        print('WARNING: Invalid mode selection')
        
        
    #Make a list of bin boundary tuples in the form (lower,upper)
    bins = list(zip(bin_inds[:-1],bin_inds[1:]))
    
    
    #Find the mean x,y coordinates of each bin.  Loop through each bin
    #division and append the mean distance to x and the mean delta to y
    x = []
    y = []
    for bin in bins:
        x.append(differences[bin[0]:bin[1]]['distance'].mean())
        y.append(differences[bin[0]:bin[1]]['delta'].mean())
        
      
    #Open a new figure and save the figure and axis handles
    fig, ax1 = plt.subplots(figsize=(13, 11), dpi= 80, facecolor='w', edgecolor='k')
    fig_vars = (fig, ax1)
        
    #Plot the raw data points
    ax1.plot(differences['distance'],differences['delta'],marker='.',color='silver',linestyle='')
    #Plot the binned averages
    ax1.plot(x,y,'k*')
    
    
    #Add x and y labels
    ax1.set_ylabel(labels['ylabel'])
    ax1.set_xlabel(labels['xlabel'])
    #Add the title string
    titlestr = '\n{} points binned into {} bins'.format(num_values,len(bins))
    plt.title(titlestr)
    
    return fig_vars,x,y,bins

def generate_semivariogram(differences,labels,sv_mode):
    """
    Function to iteratively generate a semivariogram
    
    NOTE: requires that the development environment be set to display figures
    in an interactive window
    
    INPUTS: 
        differences - A pandas dataframe containing a column of distances and
                    'differences' (correlation, semivariance, or covariance)
        labels - The axis labels for the plot
                    labels[0] ==> xlabel
                    labels[1] ==> ylabel
        sv_mode - The binning sv_mode:
                    --> sv_mode={'bins':N ,'mode':'eq.dist'}
                        fixed number of N bins equally spaced between min
                        and max distance.  The default is to use the sqrt 
                        of the number of points
                    --> sv_mode={'bins':N ,'mode':'eq.pts'}
                        fixed number of N bins each with the same number of
                        points
                    --> sv_mode = {'bins':[0,2,4,6,8,10,50,100,180],'mode':'user.div'}
                        strictly increasing list of bin divisions.  
                        Calculations will not be performed
                        outside {min(sv_mode['bins']):max(sv_mode['bins'])}
                    --> sv_mode['bins'] = 'sqrt' 
                        Uses the square root of the number of data points as
                        the number of bins
                    
                    NOTE: User bin divisions are in units of the distances
                    matrix
    
    OUTPUTS
        x - The average distances of the points in each bin
        y - The average difference of the points in each bin
    """
    
    #Keep running until the user quits
    makeplots=True
    while makeplots:
        #Plot the semivariogram with the current settings
        fig_vars,x,y,bins = plot_semivariogram(differences,labels,sv_mode)
        
        #Show, pause, and draw the plot (required to show the plot while
        #the user is deciding what to do next)
        plt.show()
        plt.pause(1)
        plt.draw()
        
        #Allow user to click points on the figure to select the next bin 
        #divisions
        getinput=True
        while getinput:
            #Flag to close the figure
            close_fig = True
            #Define a user menu
            inp_str = ['What next (see options below)?\n'+
                       '    1. \'e\' ==> exit\n'+
                       '    2. \'d\' ==> default bins (sqrt)\n'+
                       '    3. \'g\' ==> click on figure to define bin divisions\n'+
                       '                 left click - add point\n'+
                       '                 right click - accept current selection\n'+
                       '    4. \'s\' ==> save the current figure\n']
            #As the user for input
            flag = input(inp_str[0])
            
            #Exit the semivariogram generation function
            if flag == 'e':
                getinput = False
                makeplots = False
                close_fig = False #Leaves the figure open when the user exits
                
            #Re-plot the semivariogram with 'default' settings (square root
            #of N)
            elif flag == 'd':
                getinput = False
                sv_mode['bins'] = 'sqrt'
                sv_mode['mode'] = 'eq.pts'
                
            #get new bins from user clicks on the screen
            elif flag == 'g':
                getinput = False
                plt.show()
                #Get the new bins from the user clicks as a list of (x,y)
                #tuples
                newbins = plt.ginput(n=-1,timeout=-1,show_clicks=True,mouse_add=1,mouse_pop=2,mouse_stop=3)
                
                #Warn if too few divisions are selected and revert to default
                #binning method
                if len(newbins)<2:
                    print('***WARNING***\nNo Bin Divisions Specified.  Using default settings (sqrt of number of points)')
                    sv_mode['bins'] = 'sqrt'
                    sv_mode['mode'] = 'eq.pts'
                else:
                    #Extract the x-coordinate from the list of x,y coordinates
                    newbins = [tpl[0] for tpl in newbins]
                    #Check that the minimum bin boundary is less than the 
                    #minimum distance.  If this is not the case, set the 
                    #minimum distance
                    if min(newbins)>min(differences['distance']):
                        newbins.append(min(differences['distance']))
                        
                    #Check that the maximum bin boundary is greater than the 
                    #maximum distance.  If this is not the case, set the 
                    #maximum distance
                    if max(newbins)<max(differences['distance']):
                        newbins.append(max(differences['distance']))
                    #Sort the new bin divisions and put into the data structure
                    #to pass to the plotting function
                    sv_mode['bins'] = sorted(list(set(newbins)))
                    #Set the mode to user-defined
                    sv_mode['mode'] = 'user.div'
                    print('New Bin Divisions:\n{}'.format(sv_mode['bins']))
            
            #allow the user to save the figure to a filename of their choice
            elif flag == 's':
                getinput = False
                filename = input('Please enter the filename without the filetype extension\n')
        
                #Save the completed figure
                fig_vars[0].savefig(filename+'.png')
            else:
                print('***WARNING***\nInvalid Selection, try again')
           
        #Close the figure if the SV is to be re-generated.  If exit, leave
        #the figure open
        if close_fig:
            plt.close(fig_vars[0])
    
    return x,y
    
def bin_divs2inds(df,bin_divs,on_col = 'distance',remove_empty=True):
    """
    Takes a dataframe (df) with a numeric index in {integers} from 0:N, a list
    of bin divisions, and a column to divide on.  The dataframe must be sorted 
    by the column to divide on
    INPUTS:
        df = input dataframe
        bin_divs = bin divisions in the units of the column to divide on
        on_col = column to divide on
        
    OUTPUT:
        Bin index divisions
    """
    bin_inds = []
    #find the max value of the divide-on column
    max_value = df[on_col].max()
    #find the min value of the divide-on column
    min_value = df[on_col].min()
    #Find the number of bins below the data 
    num_bins_below_data = len(bin_divs[bin_divs<=min_value])
    #If there are more than 1 bin below the data, record the number and drop
    #if user specifies
    empty_bins_before = 0
    if num_bins_below_data>1:
        empty_bins_before = num_bins_below_data-1
        if remove_empty:
            bin_divs = bin_divs[empty_bins_before:]
        
    #Find the number of bins above the data
    num_bins_above_data = len(bin_divs[bin_divs>=max_value])
    #If there are more than 1 bin division above the data, record the number and drop
    #if user specifies
    empty_bins_after = 0
    if num_bins_above_data>1:
        empty_bins_after = num_bins_above_data-1
        if remove_empty:
            bin_divs = bin_divs[:-empty_bins_after]
    
    #Find the lower bounds of bins 1:N-1
    for div in bin_divs[:-1]:
        
        inds = df[df[on_col]>=div].index
        if len(inds) == 0:
            bin_inds.append(0)
        else:
            bin_inds.append(inds.min())
        
    #find the upper bound of bin N
    inds = df[df[on_col]<=bin_divs[-1]].index
    bin_inds.append(inds.max())
    
    #Find the number of bins in the raw bin list
    num_bins = len(bin_inds)
    #Extract the number of unique
    bin_inds_unique = np.unique(bin_inds)
    #Find the number of empty interior bins
    empty_bins_interior = num_bins - len(bin_inds_unique)
    if not remove_empty:
        empty_bins_interior -= empty_bins_after+empty_bins_before
    
    total_empty = empty_bins_after+empty_bins_before+empty_bins_interior
    
    #Print warning  if empty bins are found and inform user how many were removed
    if total_empty>0:
        print('***** WARNING: EMPTY BINS FOUND in bin_divs2inds.py *****')
        print('   {} bin(s) with upper bounds less than the min of the data'.format(empty_bins_before))
        print('   {} bin(s) with lower bounds greater than the max of the data'.format(empty_bins_after))
        print('   {} empty interior bin(s) (upper bound == lower bound) was found'.format(empty_bins_interior))
        if remove_empty:
            #If removal of empty bins, replace the full bin list with 
            #the unique bin indices list
            bin_inds = bin_inds_unique
            print('\n  A total of {} empty bins WERE REMOVED'.format(total_empty))
        else:
            #Inform user that empty bins were not removed
            print('\n  Empty bins were NOT removed')
  
    return bin_inds

def fit_exponential(x_vals,sill,rng,nugget):
    """
    Fits an exponential model given parameters from the semivariogram
    
    INPUTS
        x_vals - numpy array of 'x' values at which to calculate the model
        sill - the point at which the semivariogram plateaus
        rng - the point at which autocorrelation plateaus
        nugget - the y-intercept of the model. The semivariance of samples 
                collected at very short distances
    
    OUTPUTS
        expon - the exponential model
    """
    #Calculate the partial sill
    c1 = sill-nugget
    #calculate the model y-values
    expon = nugget+c1*(1-np.exp(-3*np.abs(x_vals)/rng))
    
    return expon

def fit_gaussian(x_vals,sill,rng,nugget):
    """
    Fits an gaussian model given parameters from the semivariogram
    
    INPUTS
        x_vals - numpy array of 'x' values at which to calculate the model
        sill - the point at which the semivariogram plateaus
        rng - the point at which autocorrelation plateaus
        nugget - the y-intercept of the model. The semivariance of samples 
                collected at very short distances
    
    OUTPUTS
        gauss - the gaussian model
    """
    #Calculate the partial sill
    c1 = sill-nugget
    #calculate the term in the exponential
    term = -1*(np.sqrt(3)*np.abs(x_vals)/rng)**2
    #calculate the model y-values
    gauss = nugget+c1*(1-np.exp(term))
    
    
    return gauss

def fit_spherical(x_vals,sill,rng,nugget):
    """
    Fits an spherical model given parameters from the semivariogram
    
    INPUTS
        x_vals - numpy array of 'x' values at which to calculate the model
        sill - the point at which the semivariogram plateaus
        rng - the point at which autocorrelation plateaus
        nugget - the y-intercept of the model. The semivariance of samples 
                collected at very short distances
    
    OUTPUTS
        spherical - the exponential model
    """
    #Calculate the partial sill
    c1 = sill-nugget
    #Calculate each individual term in the spherical model
    term1 = 1.5*np.abs(x_vals)/rng
    term2 = 0.5*(np.abs(x_vals)/rng)**3
    #calculate the model y-values
    spherical = nugget+c1*(term1-term2)
    
    #Find x-values that are beyond the range
    beyond_range = x_vals>rng
    #Find the indices of all points that are beyond the range
    inds_beyond_range = [i for i,v in enumerate(beyond_range) if v]
    #Replace all model y values at points beyond the range with the sill
    spherical[min(inds_beyond_range):] = sill
    
    return spherical

def plot_model_fit(data,model,labels,title):
    """
    Plots the semivariogram data and the fit model
    
    INPUTS
        data - tuple of x and y values from the semivariogram data
        model - tuple of x and y values from the fit model
        labels - x and y axis labels
        title - figure title
    """
    #Unpack the semivariogram points
    x,y = data
    #Unpack the model points
    model_x,model_y = model
    
    #open a new figure
    plt.figure()
    #Plot the binned semivariance points
    plt.plot(x,y,'*',label = 'binned data')
    #plot the model
    plt.plot(model_x,model_y,label = 'model fit')
    #Add axis labels and a figure title
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.title(title)
    #add a legend
    plt.legend()
    #Extract the name of the model fromthe title
    model_name = title.split(' ')[0]
    #save the figure
    plt.savefig('sample_{}_fit.png'.format(model_name))
    
    