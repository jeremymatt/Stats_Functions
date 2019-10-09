# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:28:26 2019

@author: jmatt
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bin_divs2inds import bin_divs2inds
import random


def semivariogram_time(variable, station_list, dist, fraction_to_keep, data1, data2, date_text,sv_mode = {'bins':'sqrt','mode':'eq.pts'}):
    """
    INPUTS: 
        variables ==> The variable names describing the data in data and data2
                      (IE: speed, dir, solar, temp, etc)
        station_list ==> A list of all station objects
                      (IE: speed, dir, solar, temp, etc)
        dist      ==> distances matrix between all sample points in the data
                      structures data1 and data2
        fraction_to_keep -  fraction_to_keep of non weather station points are randomly
                            selected for use in the semivariogram.  Used to save
                            computational time if there are a large number of
                            points from non-weather station sources (IE the
                            HRRR model)
        data1&2   ==> pandas dataframes containing the data for variables 1 & 2
                      index: timestamps of the data readings
                      column headers: station identifiers
        sv_mode      ==> The binning sv_mode:
                        --> sv_mode={'bins':'sqrt','mode':'eq.dist'}  (default)
                            fixed number of N bins equally spaced between min
                            and max distance.  The default is to use the sqrt 
                            of the number of points
                        --> sv_mode={'bins':'sqrt','mode':'eq.pts'}
                            fixed number of N bins each with the same number of
                            points
                        --> sv_mode = {'bins':[0,2,4,6,8,10,50,100,180],'mode':'user.div'}
                            strictly increasing list of bin divisions.  
                            Calculations will not be performed
                            outside {min(sv_mode['bins']):max(sv_mode['bins'])}
                        
                        NOTE: User bin divisions are in units of the distances
                        matrix
                            
    """
        
    #Skip station 0 if variable1=variable2
    if variables[0]==variables[1]:
        offset = 1
    else:
        offset = 0
        
    #If the variables are not both 'dir'
    if not ((variables[0] == 'dir') and (variables[1]=='dir')):
        #find the range of each variable
        var1_min = data1.min().min()
        var1_max = data1.max().max()
        var2_min = data2.min().min()
        var2_max = data2.max().max()
        #normalize each bariable
        data1 = 1-(var1_max-data1)/(var1_max-var1_min)
        data2 = 1-(var2_max-data2)/(var2_max-var2_min)
    
    if not fraction_to_keep==1:    
        #Identify the actual stations and retain
        realStations = [station for station in station_list if station.name.split('n')[0]=='Statio']
        #Identify 'stations' that are actually model points
        otherStations = [station for station in station_list if not station.name.split('n')[0]=='Statio']
        #find the number of stations to keep
        numToKeep = np.round(len(otherStations)*fraction_to_keep).astype(int)
        #Select a subset of non-weather station points
        station_subset = random.sample(otherStations,numToKeep)
        #Join the real stations and non-weather station lists
        station_list = list(realStations)
        station_list.extend(station_subset)
    
    #For each pair of stations
    for ind,station1 in enumerate(station_list):
        print('     Processing station {} of {}'.format(ind,len(station_list)))
        for station2 in station_list[ind+offset:]:
            #Find the row number of the station 2 in the distances matrix
            station2_dist_row = dist.columns.get_loc(station2.name)
            #Extract the distance between the two stations
            distance = dist[station1.name][station2_dist_row]
            try:
                #See if data from station 1 appears in the variable1 dataframe
                var1_data = data1[station1.name]
                #See if data from station 2 appears in the variable2 dataframe
                var2_data = data2[station2.name]
            except:
                continue
                
            #Calculate the differences between each of the variables.
            diff_temp = pd.DataFrame({'distance':distance,'delta':np.abs(var1_data-var2_data)})
            #Drop all NaN values
            diff_temp.dropna(inplace=True)
            try: differences = differences.append(diff_temp,ignore_index=True)
            except: differences = diff_temp
            
    
    #Column to split bin divisions on
    on_col = 'distance'
    #Sort the dataframe ascending
    differences.sort_values(on_col,ascending=True,inplace=True)
    #Reset the index, dropping the old index values
    differences = differences.reset_index(drop=True)
    #Determine the number of values being plotted
    numValues = differences.shape[0]
    makeplots = True
    while makeplots:
    
        if sv_mode['bins'] == 'sqrt':
            sv_mode['bins'] = np.round(numValues**0.5).astype(int)
          
        #Find the indices of the bin divisions depending on the mode
        if sv_mode['mode'] == 'eq.dist':
            #Find the min and max distances
            minDist = np.floor(differences['distance'].min())
            maxDist = np.ceil(differences['distance'].max())
            #Generate the bin divisions in terms of distances
            bin_divs = np.linspace(minDist,maxDist,sv_mode['bins']+1)
            #Convert distances to indices
            bin_inds = bin_divs2inds(differences,bin_divs,on_col,remove_empty=True)
        elif sv_mode['mode'] == 'eq.pts':
            #Break the points up into bins each containing approximately equal
            #numbers of points
            
            bin_inds = np.round(np.linspace(0,differences.shape[0],sv_mode['bins']+1)).astype(int)
        elif sv_mode['mode'] == 'user.div':
            #Convert the user-provided distance bins into indices
            bin_inds = bin_divs2inds(differences,np.array(sv_mode['bins']),on_col,remove_empty=False)
        else:
            print('WARNING: Invalid mode selection')
            
        #Make a list of bin boundary tuples
        bins = list(zip(bin_inds[:-1],bin_inds[1:]))
        
        #Find the mean x,y coordinates of each bin
        x = []
        y = []
        for bin in bins:
            x.append(differences[bin[0]:bin[1]]['distance'].mean())
            y.append(differences[bin[0]:bin[1]]['delta'].mean())
            
        #Find the maximum delta
        max_delta = np.max(y)
            
        #Plot histograms of the tweet/hr counts
        fig, ax1 = plt.subplots(figsize=(13, 11), dpi= 80, facecolor='w', edgecolor='k')
        ax1.plot(x,y,'b*')
        
        #Plot bin division boundaries
    #    for ind in bin_inds:
    #        ax1.plot([differences.iloc[ind]['distance'],differences.iloc[ind]['distance']],[0,max_delta],'y--')
        
        #Add x and y labels
        ax1.set_ylabel('mean $\Delta$')
        ax1.set_xlabel('Distance')
        #Build a string for the title
        titlestr1 = 'Semivariogram of date range: {}'.format(date_text)
        titlestr2 = '\nVar1: {} (range:{:.3f} to {:.3f})'.format(
                variables[0].split(':')[0],
                var1_min,
                var1_max)
        titlestr3 = '\nVar2: {} (range:{:.3f} to {:.3f})'.format(
                variables[1].split(':')[0],
                var2_min,
                var2_max)
        titlestr4 = '\n{} points binned into {} bins'.format(numValues,len(bins))
        if (variables[0]=='dir:') and (variables[1]=='dir:'):
            normStyle = '\n   Interior angle calculated and not normalized'
        else:
            normStyle = '\n   Variables normalized to 0-1 before subtraction'
        #Add the title string
        plt.title(titlestr1+titlestr2+titlestr3+titlestr4+normStyle)
        #Add gridlines
        plt.grid()
        plt.grid(which='minor')
        #Add the legend
    #    ax1.legend(prop={'size': font_size})
        font_size = 20
        #Resize the fonts for all items in axis 1
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(font_size)
            
        
        
        plt.show()
        plt.pause(0.1)
        
        getinput=True
        while getinput:
            inp_str = ['What next (see options below)?\n'+
                       '    1. \'e\' ==> exit\n'+
                       '    2. \'d\' ==> default bins (sqrt)\n'+
                       '    3. \'g\' ==> click on figure to define bin divisions\n'+
                       '    4. \'s\' ==> save the current figure\n']
            
            flag = input(inp_str[0])
            
            if flag == 'e':
                getinput = False
                makeplots = False
            elif flag == 'd':
                getinput = False
                sv_mode['bins'] = 'sqrt'
                sv_mode['mode'] = 'eq.pts'
            elif flag == 'g':
                getinput = False
                plt.show()
                plt.pause(0.1)
                newbins = plt.ginput(n=-1,timeout=-1,show_clicks=True,mouse_add=1,mouse_pop=2,mouse_stop=3)
                
                if len(newbins)<2:
                    print('***WARNING***\nNo Bin Divisions Specified.  Using default settings (sqrt of number of points)')
                    sv_mode['bins'] = 'sqrt'
                    sv_mode['mode'] = 'eq.pts'
                else:
                    
                    newbins = [tpl[0] for tpl in newbins]
                    breakhere=1
                    
                    if min(newbins)>min(differences['distance']):
                        newbins.append(min(differences['distance']))
                    breakhere=1
                    if max(newbins)<max(differences['distance']):
                        newbins.append(max(differences['distance']))
                    breakhere=1
                    sv_mode['bins'] = sorted(list(set(newbins)))
                    sv_mode['mode'] = 'user.div'
                    print('New Bin Divisions:\n{}'.format(sv_mode['bins']))
            elif flag == 's':
                getinput = False
                filename = input('Please enter the filename without the filetype extension\n')
        
                #Save the completed figure
                fig.savefig(filename+'.png')
            else:
                print('***WARNING***\nInvalid Selection, try again')
                
    
        plt.close(fig)

