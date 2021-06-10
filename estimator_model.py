# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:33:15 2020

@author: Parisa

This program is used for building the estimator model. We considered building 
model for each learning materials that is a model for country-capital easy deck,
country-capital hard, numerical.

First, the program load the pre-process data file (pickle files), then separate
 each column into variables. Next, we applied log-normalization followed by
principal component analysis (PCA) to the retained data. After that, in order to
 increase the number of data points, we propose using the synthetic minority 
 oversampling technique (SMOTE).

After finished the preparation of data, we applied Naive Bayes as an algorithm 
for our model and test the accuracy of the models. Lastly, we save the model into
 the pickle files for use in the e-learning software later.


"""
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV 

from importlib import reload
import preferences
reload(preferences)
import constants
reload(constants)

reload(constants)



def load(m, d):
    """
    load prepared data (pickle file) 
    """
    fname_out = 'pkl_vars/'+m+'_rf_'+d+'_admissable.pkl'
    with open(fname_out, 'rb') as handle:
        bvars = pickle.load( handle )
        
    return bvars

def log_pp(feat1, feat2):
    """    
    log transformation, function return the list of data which applied log transformation.
    
    y = log(x)
    """
    tempx_pp = list( np.log(feat1) )
    tempy_pp = list( np.log(feat2) )
    
    return tempx_pp, tempy_pp

def prep_for_pca(feat1, feat2):
    """
    prepared data for fetch into PCA.
    """
    feats = []
    feats = [[f1, f2] for f1, f2 in zip(feat1, feat2)]
    return feats


if __name__ == "__main__":
    
    start_time = time.time()
    
    for m in constants.MEDIA: 
        """
        m is media types which are either v (visual), a (audio), av (audiovisual). 
        In this study, we used only the v (visual) media type to be input for
        building the estimator model.
        """
        if m=='av' or m=='a':
            continue
        for d in constants.DECKS:
            if d in preferences.DECKS_OMITTED:
                continue
            
            # Load raw data and separate each column into variables.
            temp_bvars_agr_before = load(m,d)
            feat1_raw = temp_bvars_agr_before[preferences.MYBV1]
            feat2_raw = temp_bvars_agr_before[preferences.MYBV2]
            labels_raw = temp_bvars_agr_before['rf']
            
            # prepocessing with log transformation
            feat1_prepro, feat2_prepro = log_pp(feat1_raw, feat2_raw)
            
            # prepare the data for PCA 
            feats = prep_for_pca(feat1_prepro, feat2_prepro)
            
            #split raw data to train set(80%) and test set(20%) 
            feats_train, feats_test, labels_train, labels_test = train_test_split(feats, labels_raw,\
                                                                                      test_size=0.2, \
                                                                                     random_state=12345)            
            
            #apply PCA to train set and transform the test set to matching with train set
            pca = PCA(n_components=2).fit(feats_train)
            feats_train_transformed = pca.transform(feats_train)
            feats_test_new = pca.transform(feats_test)
            
            
            # create synthetic point to increase the number of data point for minor class by SMOTE 
            sm = SMOTE(sampling_strategy = 'auto',k_neighbors = preferences.NUM_K_SMOTE , random_state = 12345)
            feats_train_final, labels_train_final = sm.fit_resample(feats_train_transformed,labels_train)
            
            
            #apply Naive bayes and train the estimator model.
            params = {}
            model = GridSearchCV(GaussianNB(),params,verbose = 1,cv = 3,n_jobs = -1)
            model_results = model.fit(feats_train_final,labels_train_final) 
            print(model_results)
            
            #use the test set data to test the estimator model.
            y_pred = model_results.predict(feats_test_new)
            
            #report the confusion matrix and classification report for checking the quality of model.
            print(confusion_matrix(labels_test, y_pred))
            print(classification_report(labels_test, y_pred))
            
            #save model into pickle file which can use as a estimator model in another software.
            pca_model = 'pre_model_pca_'+d+'.pkl'
            model_filename = 'rf_model_naive_'+d+'.pkl'
            
            
            elapsed_time = time.time() - start_time
            print('Time elapsed %2.2f sec' %elapsed_time)
            

      