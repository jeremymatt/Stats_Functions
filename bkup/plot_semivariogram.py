# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:40:36 2019

@author: jmatt
"""
import numpy as np
import matplotlib.pyplot as plt
from bin_divs2inds import bin_divs2inds
from cycler import cycler

def plot_semivariogram(fig_vars,sv_mode,numValues,diff_to_plot,on_col,title_str):
    
    if sv_mode['bins'] == 'sqrt':
        sv_mode['bins'] = np.round(numValues**0.5).astype(int)
      
    #Find the indices of the bin divisions depending on the mode
    if sv_mode['mode'] == 'eq.dist':
        #Find the min and max distances
        minDist = np.floor(diff_to_plot['distance'].min())
        maxDist = np.ceil(diff_to_plot['distance'].max())
        #Generate the bin divisions in terms of distances
        bin_divs = np.linspace(minDist,maxDist,sv_mode['bins']+1)
        #Convert distances to indices
        bin_inds = bin_divs2inds(diff_to_plot,bin_divs,on_col,remove_empty=True)
    elif sv_mode['mode'] == 'eq.pts':
        #Break the points up into bins each containing approximately equal
        #numbers of points
        
        bin_inds = np.round(np.linspace(0,diff_to_plot.shape[0],sv_mode['bins']+1)).astype(int)
    elif sv_mode['mode'] == 'user.div':
        #Convert the user-provided distance bins into indices
        bin_inds = bin_divs2inds(diff_to_plot,np.array(sv_mode['bins']),on_col,remove_empty=False)
    else:
        print('WARNING: Invalid mode selection')
        
    #Make a list of bin boundary tuples
    bins = list(zip(bin_inds[:-1],bin_inds[1:]))
    
    Station_name = diff_to_plot.keys()[-1]
    diff_to_plot.rename(columns={Station_name:'delta'},inplace=True)
    
    #Find the mean x,y coordinates of each bin
    x = []
    y = []
    for bin in bins:
        x.append(diff_to_plot[bin[0]:bin[1]]['distance'].mean())
        y.append(diff_to_plot[bin[0]:bin[1]]['delta'].mean())
     
    #Find the larges y-value
    y_max = np.ceil(max(y))
     
    if fig_vars=='newfig':
        fig, ax1 = plt.subplots(figsize=(13, 11), dpi= 80, facecolor='w', edgecolor='k')
        fig_vars = (fig, ax1, y_max)
        default_cycler = (cycler(color=['b','r','k','c','g','b','r','k','c','g','b','r','k','c','g']) +
                  cycler(marker=['.','v','^','<','>','1','2','3','4','s','p','P','*','+','x'])+
                  cycler(linestyle=['','','','','','','','','','','','','','','']))
        plt.rc('axes', prop_cycle=default_cycler)
        #Assign the cycler to each axes
        ax1.set_prop_cycle(default_cycler)
    else:
        fig, ax1, temp = fig_vars
        fig_vars = (fig, ax1, y_max)
      
    ax1.plot(diff_to_plot['distance'],diff_to_plot['delta'],label=Station_name+'-Raw',marker='.',color='silver',linestyle='')
    ax1.plot(x,y,label=Station_name)
    
    #Plot bin division boundaries
#    for ind in bin_inds:
#        ax1.plot([diff_to_plot.iloc[ind]['distance'],diff_to_plot.iloc[ind]['distance']],[0,max_delta],'y--')
    
    #Add x and y labels
    ax1.set_ylabel('mean $\Delta$')
#    ax1.set_ylim(0,175)
    ax1.set_xlabel('$\Delta$ Hours')
    #Add the title string
    
    titlestr4 = '\n{} points binned into {} bins'.format(numValues,len(bins))
    title_str=title_str+titlestr4
    plt.title(title_str)
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
        
    return fig_vars