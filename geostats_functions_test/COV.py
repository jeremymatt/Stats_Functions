# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:22:54 2019

@author: jmatt
"""

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
    cov = (data[variables[0]]-means[variables[0]])*(data[variables[1]]-means[variables[1]])
    #Normalize by N
    cov = cov.mean()
    
    return cov