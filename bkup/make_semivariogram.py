# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:01:55 2018

@author: jmatt
"""

import numpy as np
import pandas as pd
import numbers as num

from semivariogram import semivariogram
from semivariogram_time import semivariogram_time
from select_daterange import select_daterange
from IdahoDST_to_UTC import IdahoDST_to_UTC

def make_semivariogram(save_fn,ALLdata,sv_type,dist,fraction_to_keep,variables,date_range,mode,station_sets,stations='all',sv_mode = {'bins':25,'mode':'eq.dist'} ):
    """
    Driver to make semivariograms from the weather data.  Using the input date
    range and variables, data structure, etc, this function extracts and formats
    the variable data that needs to be sent to the semivariogram function
    
    INPUTS:
        ALLdata - the weather data object
        stations - list of stations to include in the calculation (default = 'all')
        dist - the distances matrix
        fraction_to_keep -  fraction_to_keep of non weather station are randomly
                            selected for use in the semivariogram.  Used to save
                            computational time if there are a large number of
                            points from non-weather station sources (IE the
                            HRRR model)
        variables - the variables to generate the semivariogram for
        date_range - the range of times over which to calculate the semivariogram
                    data_range = [Start,End], with Start/End having format of
                    integer year, month, day
        mode - The select mode:
                hard ==> hard date range between the dates specified
                seasonal ==> will calculate between mon/day1 and mon/day2 for all years
        sv_mode      ==> The binning sv_mode:
                        --> sv_mode={'bins':10,'mode':'eq.dist'}  (default)
                            fixed number of N bins equally spaced between min
                            and max distance
                        --> sv_mode={'bins':10,'mode':'eq.pts'}
                            fixed number of N bins each with the same number of
                            points
                        --> sv_mode = {'bins':[0,2,4,6,8,10,50,100,180],'mode':'user.div'}
                            strictly increasing list of bin divisions.  
                            Calculations will not be performed
                            outside {min(sv_mode['bins']):max(sv_mode['bins'])}
                        
                        NOTE: User bin divisions are in units of the distances
                        matrix
    """
    
    var1_headers = ['datetime_bins',variables[0]]
    var2_headers = ['datetime_bins',variables[1]]
    
    #Generate a list of pointers to the datasets for each station
    station_list = []
    if stations == 'all':
        station_list = [ALLdata.WSdata[x] for x in range(0,ALLdata.numFiles)]
    else:
        for x in stations:
            if isinstance(x,num.Number):
                station_list.append(ALLdata.WSdata[x])
            else:
                station_list.append(ALLdata.WSdata[ALLdata.StationNamesIDX[x]])
    
    #If data1/data2 exist, delete them
    try: del data1
    except: pass

    #For each station in the list, grab the variable1 and variable2 data columns
    #and store in data1 and data2 dataframes
    data1 = pd.DataFrame()
    for station in station_list:
        breakhere=1
        #Try to get the data; if it doesn't exist, then pass
        try:
            #Extract the variable1 data for the current station and set the index
            #to the datetime column
            newdata = station.data_binned[var1_headers].set_index('datetime_bins')
            #Change the data column name to the station name (to make each col name
            #unique)
            newdata.columns = [station.name]
            #Add the data column for the current station to the datablock
            #for variable 1
            data1 = pd.concat([data1,newdata],axis=1)
        except: pass
    
    #Keep only the data that is in the specified range for the specified range
    #mode
    data1 = select_daterange(data1,date_range[0],date_range[1],mode)
        
#    If either of the variables is 'solar', keep only the data between 6AM and 
#    6PM
#    if (variables[0] == 'solar:') or (variables[1] == 'solar:'):
#        data1 = data1.between_time('8:00','16:00')
        
    
    #Generate a string for labeling the semivariogram figure
    if mode == 'hard':
        start_str = '{}-{}-{}'.format(date_range[0][0],date_range[0][1],date_range[0][2],)
        end_str = '{}-{}-{}'.format(date_range[1][0],date_range[1][1],date_range[1][2],)
    elif mode == 'seasonal':
        start_str = '{}-{}'.format(date_range[0][1],date_range[0][2],)
        end_str = '{}-{}'.format(date_range[1][1],date_range[1][2],)
    else:
        start_str = 'MODE ERROR'
        end_str = 'MODE ERROR'
        print('WARNING: invalid mode')  
    date_text = 'Date range ({}):{} through {}'.format(mode,start_str,end_str)
    
    
    differences = semivariogram(save_fn,station_sets,sv_type, variables, station_list, dist, fraction_to_keep, data1, date_text,sv_mode)
    
    return data1,station_list,differences