# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:42:33 2019

@author: jmatt
"""

def calc_difference(row,row_values,variable):
    
    breakhere=1
    if variable == 'dir:':
        keys = row.keys()[1:]
        diff=row-row_values
        diff[keys] = abs(diff[keys])
        mask = diff[keys]>180
        corrected = 360-diff[keys][mask]
        keys_corr = corrected.keys()
        diff[keys_corr] = corrected
    else:
        diff = row-row_values
        
    return diff