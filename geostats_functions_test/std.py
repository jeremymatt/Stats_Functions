# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:23:45 2019

@author: jmatt
"""

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