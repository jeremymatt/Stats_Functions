# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:59:24 2018

@author: jmatt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bin_divs2inds import bin_divs2inds
from plot_semivariogram import plot_semivariogram
from calc_difference import calc_difference
import random


def semivariogram(save_dir,station_sets,sv_type,variables, station_list, dist, fraction_to_keep, data1, date_text,sv_mode = {'bins':'sqrt','mode':'eq.pts'}):
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
        
    try: del differences
    except: pass

    data1_keys = data1.keys()  
    
    #If the variables are not both 'dir'
    if not ((variables[0] == 'dir') and (variables[1]=='dir')):
        #find the range of each variable
        var1_min = data1[data1_keys].min().min()
        var1_max = data1[data1_keys].max().max()
        #normalize each bariable
#        data1[data1_keys] = 1-(var1_max-data1)/(var1_max-var1_min)
    
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
    
    if sv_type == 'distance':
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
                    var2_data = data1[station2.name]
                except:
                    continue
                    
                #Calculate the differences between each of the variables.
                diff_temp = pd.DataFrame({'distance':distance,'delta':np.abs(var1_data-var2_data)})
                #Drop all NaN values
                diff_temp.dropna(inplace=True)
                try: differences = differences.append(diff_temp,ignore_index=True)
                except: differences = diff_temp
    else:
        data1.reset_index(inplace=True)
        for row in range(data1.shape[0]):
            print(row)
            row_values = data1.iloc[[row]].values[0]
            next_rows = pd.DataFrame(data1.iloc[row+1:,:])
            temp = next_rows.apply(lambda row: calc_difference(row,row_values,variables[0]), axis=1)
            try: differences=differences.append(temp)
            except: differences = pd.DataFrame(temp)
        
        differences[data1_keys]=differences[data1_keys].abs()
        differences.rename(columns={'datetime_bins':'distance'},inplace=True)
        differences['distance']=differences['distance']/np.timedelta64(1, 'h')
        
    
    #Column to split bin divisions on
    on_col = 'distance'
    #Sort the dataframe ascending
    differences.sort_values(on_col,ascending=True,inplace=True)
    #Reset the index, dropping the old index values
    differences = differences.reset_index(drop=True)
    #Determine the number of values being plotted
    numValues = differences.shape[0]
    makeplots = True
    
    
#    #Build a string for the title
#    titlestr1 = 'Semivariogram of date range: {}'.format(date_text)
#    titlestr2 = '\n for variable: {} (un-normalized range:{:.3f} to {:.3f})'.format(
#            variables[0].split(':')[0],
#            var1_min,
#            var1_max)
#    station_names=[station.name.split('ion')[1] for station in station_list]
#    titlestr3 = '\nStations: {}'.format(station_names)
#        
#    title_str = titlestr1+titlestr2+titlestr3
    
    for station_list in station_sets:
        
        
        save_fn = '{}\\{}\\{}-{}'.format(save_dir,variables[0].split(':')[0],station_list[0].split('ion')[1],station_list[-1].split('ion')[1])
        
        
        #Build a string for the title
        titlestr1 = 'Semivariogram of date range: {}'.format(date_text)
        titlestr2 = '\n for variable: {} (un-normalized range:{:.3f} to {:.3f})'.format(
                variables[0].split(':')[0],
                var1_min,
                var1_max)
        station_names=[station.split('ion')[1] for station in station_list]
        titlestr3 = '\nStations: {}'.format(station_names)
            
        title_str = titlestr1+titlestr2+titlestr3
        
        auto=True
        if not auto:
            while makeplots:
                fig_vars='newfig'
                for station in station_list:
                    diff_to_plot = pd.DataFrame(differences[['distance',station]])
                    fig_vars = plot_semivariogram(fig_vars,sv_mode,numValues,diff_to_plot,on_col,title_str)
                 
                fig_vars[1].legend(prop={'size': 16})
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
                        fig_vars[0].savefig(filename+'.png')
                    else:
                        print('***WARNING***\nInvalid Selection, try again')
                        
            
                plt.close(fig_vars[0])
            
            
        else:
        
            fig_vars='newfig'
            for station in station_list:
                diff_to_plot = pd.DataFrame(differences[['distance',station]])
                fig_vars = plot_semivariogram(fig_vars,sv_mode,numValues,diff_to_plot,on_col,title_str)
             
            y_max = {}
            y_max['dir:']=180
            y_max['speed:'] = 16
            y_max['temp:'] = 22
            y_max['solar:'] = 700
            
            max_diff = fig_vars[2]
            y_max_val = int(max([y_max[variables[0]],max_diff]))
            print('max_diff:{}, hard_code:{}, selected:{}'.format(max_diff,y_max[variables[0]],y_max_val))
            
            fig_vars[1].set_ylim(0,y_max_val)
            
            fig_vars[1].legend(prop={'size': 16})
              
            fig_vars[0].savefig(save_fn+'.png')
        
            plt.close(fig_vars[0])


    return differences
