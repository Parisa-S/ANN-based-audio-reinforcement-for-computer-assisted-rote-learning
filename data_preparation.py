#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:10:50 2020

@author: zeynep

This function computes the cdf of tc, tq, ta and rqa.

Based on each of those, it defines a threshold value, which bounds the lower 
95% of the relating data. 

I consider the data points, which has all its fields below relating threshols,
as admissable. Namely, if all tc, tq, ta and rqa are below the threshold, I 
consider that data point as admissable. If any one of them is above the threshold, 
I discard that data point. 

The function then clones the data by copying only the admissable data points. 

In addition, while cloning, I remove the data points which have remember_or_forget 
as -1. That means the card is removed from the deck of other participants. So 
there is no point in considering that card or data point.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from importlib import reload
import preferences
reload(preferences)
import constants
reload(constants)


"""
Columns are :
    col0: tc
    col1: tq
    col2: ta
    col3: rqa
    col4: vprime
    col5: remember_or_forget_or_invalid (1/0/-1)
"""

def init():
    bvars_agr, bvars_agr_admissable = {}, {}
    for bv in constants.BEHAV_VARS:
        bvars_agr[bv], bvars_agr_admissable[bv] = [], []
       
    return  bvars_agr, bvars_agr_admissable 
    
def load(bvars_agr):
    for m in constants.MEDIA:
        for d in constants.DECKS:
            fname = '../data/behavioral_variables/'+m+'_rf_'+d+'.txt'
            data = np.loadtxt(fname, dtype='f', delimiter=' ')

            for i, bv in enumerate(constants.BEHAV_VARS):
                temp_col = [r[i] for r in data]
                bvars_agr[bv].extend( temp_col )
                
    return bvars_agr
      
def get_thresholds(bvars_agr):
    """
    Get thresholds for MAX_PERC precentile 
    """
    thresh_max_perc, bv_hists = {}, {}
    for bv in constants.BEHAV_VARS:
        
        if bv == 'vprime' or bv == 'rf':
            continue
        
        
        bv_hists[bv], bin_edges = np.histogram(bvars_agr[bv], \
                bins = preferences.NBINS_HIST, \
                density=True)
        
        # the first index where it passes 
        bin_size = bin_edges[1]-bin_edges[0]
        indx = np.min(\
                      np.where(\
                               np.cumsum(bv_hists[bv]) * bin_size > preferences.MAX_PERC_ADMISSABLE\
                               ))
        
        bin_edges_midpt = np.multiply(0.5, np.add(bin_edges[0:-1], bin_edges[1:]))
        thresh_max_perc[bv] = bin_edges_midpt[indx]
        
    #    plt.figure()
    #    plt.plot(bin_edges_midpt, bv_hists[bv])
    #    plt.title(bv)
    #    plt.grid(linestyle='dotted')
    #    plt.show()
    return thresh_max_perc
    
      
def clone_admissable(thresh_max_perc, bvars_agr):
    """
    Clone the data by removing any data point which:
        has at least one field that belongs to more than MAX_PERC precentile 
        does not have an rf of 0 or 1 (-1 means card is removed)
        
    """
    
    for m in constants.MEDIA:
        print('---------------------')
        for d in constants.DECKS:
            
            bvars_old, bvars_new = {}, {}
            
            for bv in constants.BEHAV_VARS:
                bvars_old[bv], bvars_new[bv] = [], []
            
            fname = '../data/behavioral_variables_100/'+m+'_rf_'+d+'.txt'
            data = np.loadtxt(fname, dtype='f', delimiter=' ')
            
            for i, bv in enumerate(constants.BEHAV_VARS):
                temp_col = [r[i] for r in data]
                bvars_old[bv].extend( temp_col )
            
            
            
            for i, (tc0, ta0, tq0, rqa0, rf0) in \
            enumerate(zip(bvars_old['tc'], bvars_old['tq'], bvars_old['ta'],bvars_old['rqa'], bvars_old['rf'])):
                if (tc0 < thresh_max_perc['tc'] and\
                    ta0 < thresh_max_perc['ta'] and\
                    tq0 < thresh_max_perc['tq'] and\
                    rqa0 < thresh_max_perc['rqa'] and\
                    rf0  > -1 ):
        
                    for bv in constants.BEHAV_VARS:
                        bvars_new[bv].append(bvars_old[bv][i] )
                    
            fname_out = 'pkl_vars/'+m+'_rf_'+d+'_admissable.pkl'
            with open(fname_out, 'wb') as handle:
                pickle.dump(bvars_new, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                
            print('{}_{} old_size {} new_size {}'.format(m, d, \
                  len(bvars_old['tc']),\
                  len(bvars_new['tc'])))
    
        
    return bvars_new

        
###############################################################################    
if __name__ == "__main__":
    
    start_time = time.time()
    
    bvars_agr, bvars_agr_admissable  = init()
    bvars_agr = load(bvars_agr)
    thresh_max_perc = get_thresholds(bvars_agr)
    bvars_agr_admissable = clone_admissable(thresh_max_perc, bvars_agr)  
    #display_hist(bvars_agr_admissable)
    

    # 0.50 sec
    elapsed_time = time.time() - start_time
    print('\nTime elapsed %2.2f sec' %elapsed_time)          