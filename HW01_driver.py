# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from geostats_functions import cov
from geostats_functions import rho
from geostats_functions import extract_pairs


#load data
filename = 'Burea_subset.xls'
x = 'x (mm)'
y = 'y (mm)'
res = 'Water Resisitivity ohm-m'
perm = 'Permeability (mD)'
data= pd.read_excel(filename)

#data = data[:5]

#Problem 1
plt.plot(data[res],data[perm],'.')
plt.xlabel(res)
plt.ylabel(perm)

variables = [res,perm]

covariance = cov(data,variables)
pearson_rho = rho(data,variables)

pairs = extract_pairs(data,x,y,variables)
