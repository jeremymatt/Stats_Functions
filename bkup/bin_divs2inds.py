# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:21:37 2018

@author: jmatt
"""

import numpy as np
import pandas as pd

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
    #If there are more than 1 bin above the data, record the number and drop
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