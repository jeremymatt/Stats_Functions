# -*- coding: utf-8 -*-
import pandas as pd

def cov(data,variables):
    """
    Calculates the covariance between two variables
    
    INPUTS
        data - pandas dataframe containing the data
        variables - list of the variable names
        
    OUTPUT
        covariance
        
    """
    
    means = data[variables].mean()

    cov = (data[variables[0]]-means[variables[0]])*(data[variables[1]]-means[variables[1]])
    cov = cov.mean()
    
    return cov

def std(data,variable):
    """
    calculates the standard deviation 
    """
    mean = data[variable].mean()
    std = (data[variable]-mean)**2
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
    
    covariance = cov(data,variables)
    std1 = std(data,variables[0])
    std2 = std(data,variables[1])
    
    rho = covariance/(std1*std2)
    return rho
    
def select_lag(data,x,y,lag,theta):
    """
    Returns a subset based on a lag and angle
    """
    deltas = pd.DataFrame()
    for i in range(len(data)):
        temp = pd.DataFrame()
        cur_x = data.loc[i,x]
        cur_y = data.loc[i,y]
        temp['dx'] = data.loc[i+1:,x]-cur_x
        temp['dy'] = data.loc[i+1:,y]-cur_y
        deltas.append(temp)
    
    