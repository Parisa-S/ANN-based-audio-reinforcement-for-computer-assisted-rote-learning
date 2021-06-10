#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:22:26 2020

@author: zeynep
"""

NBINS_HIST = 100
"""
MAX_PERC_ADMISSABLE is between 0 and 1. 
I accept the data upto the this percentile

0.9545 corresponds to 2*sigma

Check https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
"""
MAX_PERC_ADMISSABLE = 0.9545 

#MY_TEMP_NBINS_HIST = 8

#NUM_K_NEIGHBOR = 5
NUM_K_SMOTE = 9

GRID_PARAMS_KNN = {
        'n_neighbors': [5,7,9,11,13,15,17,19],
        'weights': ['distance'],
        'metric' : ['euclidean','manhattan']
}

GRID_PARAMS_SVM = {
        'C': [0.1,1, 10, 100], 
        'gamma': [0.001,0.01,0.1,1],
        'kernel': ['rbf', 'poly', 'sigmoid']
}

GRID_PARAMS_RDF = {
    'n_estimators'      : [100,200,500,700,1000,1200,1500,2000],  #number of trees in the foreset
    'max_depth'         : [5,10,20,50,100], #max number of levels in each decision tree
    'random_state'      : [0],
    'max_features'      : ['auto','sqrt'], #max number of features considered for splitting a node
}

GRID_PARAMS_NB = {}
"""
if you want omit a deck:
    e
    h
    n
"""
DECKS_OMITTED = []
"""
if you want omit, a medium (or some media)
"""
MEDIA_OMITTED = ['a', 'av']

"""
The behavioral variables that I consider in my analysis
"""
MYBV1 = 'rqa'
MYBV2 = 'ta'

