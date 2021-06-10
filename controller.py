# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:39:31 2020

@author: Momo
"""
import pickle
import numpy as np
import time
import glob
import os

#initial variable
start_time = ''
flip_time = ''
ans_time = ''
model = ''
card_id = ''
data = dict()
file = ''
modal = ''
difficulty = ''
r_ratio = 0
predict_result = []


    
def deck_type(deck_name):
    #name_format : deckdiff_modal
    """
    convert the deck_name into code for controller.
    """
    global modal,difficulty
    modal = deck_name[8]
    difficulty = deck_name[3].lower()

    return difficulty,modal

    
def set_starttime(time):
    """
    set the time_start into variable
    """
    global start_time
    start_time = time
    print('start_time: {}'.format(start_time))
    
def set_fliptime(time):
    """
    set the time_flip into variable
    """
    global flip_time
    flip_time = time
    print('flip_time: {}'.format(flip_time))
    
def set_anstime(time):
    """
    set the time_evaluation into variable
    """
    global ans_time
    ans_time = time
    print('ans_time: {}'.format(ans_time))
    
def set_cardID(cid):
    """
    set the card_id into variable
    """
    global card_id
    card_id = cid
    print('card_id: {}'.format(card_id))
    
def set_deckID(did):
    """
    set the deck_id into variable
    """
    global deck_id
    deck_id = did
    print('deck_id: {}'.format(deck_id))
    
def set_deckinfo(cids,d_name):
    """
    set the deck_info into variable
    """
    global deck_name ,data
    deck_name = d_name
    #cids = cids
    data = dict((key,[1]) for key in cids)
    print(data)

def set_profile_name(name):
    """
    set the profile_name into variable
    """
    global profile_name
    profile_name = name
    create_directory(profile_name)

def create_directory(name):
    """
    if the directory is not exists, create the new directory for storing 
    the log files
    """
    located_path = 'D:/Okayama U/2020-4 research/2020_09_Anki/model/log/'
    path_name = name
    
    try:
        os.mkdir(located_path+path_name)
    except OSError:
        print("Failed to create new folder")


def set_rratio(ratio):
    """
    set the audio_trigger_ratio into variable
    """
    global r_ratio
    r_ratio = ratio
    print('r_ratio: {}'.format(r_ratio)) 
    
def load_model(deck):
    """
    load the model which match to the deck types.
    this function will called from Anki when open the deck.
    """
    deck_name,modal = deck_type(deck) 
    if modal == 'R':
        read_ratio_from_estm()
    print(modal)
    if deck_name != None:
        model_filename = 'D:/Okayama U/2020-4 research/2020_09_Anki/model/rf_model_naive_'+ deck_name +'.pkl'
        pre_model_filename = 'D:/Okayama U/2020-4 research/2020_09_Anki/model/pre_model_pca_'+ deck_name +'.pkl'
        with open(model_filename,'rb') as model_file:
            global model_naive
            model_naive = pickle.load(model_file)
        with open(pre_model_filename,'rb') as pre_model_file:
            global model_pca
            model_pca = pickle.load(pre_model_file)
    else:
        print('Error. Can not load model')

def predict():
    """
    This function is use for predicting the learners' likelihood of remembering
    or forgetting. First, we calculate the behavioral variable used as an input
    that is tq, ta, rqa. 
    
    If the modality is Estimation, we applied log-transormation and PCA to 
    transform the data before use it. Then use the loaded model to predict it.
    Last, we update the audio trigger log file and also store the result into 
    a dict which is a temp variable in the program.
    
    If the modality is Random, we random the number between 0 and 1 then check 
    the trigger rate ratio and convert it to the result as estimation method
    (e.g.forget or remember). Last, we update the audio trigger log file and also
    store the result into a dict which is a temp variable in the program.
    
    If the modality is Visual, we set the predicted result to 1 (i.e. remember).
    Then, we update the audio trigger log file and also store the result into a
    dict which is a temp variable in the program.
    
    
    """
    global new_feats
    tq = (flip_time-start_time)*1000
    ta = (ans_time-flip_time)*1000
    rqa = tq/ta
    
    print('tq{} ta{} rqa{}'.format(tq,ta,rqa))
    print(modal)
    if modal == 'E':
        print('using Estimation')
        #log transform
        log_rqa = np.log(rqa)
        log_ta = np.log(ta)
        
        #pca
        feats = np.array([log_rqa,log_ta],dtype=np.float64)
        new_feats = np.reshape(feats,[1,2])
        print(new_feats)
        
        pca_feats = model_pca.transform(new_feats)
        
        y_pred = model_naive.predict(pca_feats)
        
        update_dict(int(y_pred[0]))
        predict_result.append(y_pred[0])
        write_log(card_id,tq,ta,rqa,y_pred[0])
        
    elif modal == 'R':
        print('using Random')
        ran_num = np.random.random()
        print(ran_num)
        print(r_ratio)
        if ran_num > float(r_ratio):
            y_pred = 1
        else:
            y_pred = 0
        
        update_dict(y_pred)
        predict_result.append(y_pred)
        write_log(card_id,tq,ta,rqa,y_pred)
        
    else:
        print('using visual')
        y_pred = 1
        
        update_dict(y_pred)
        predict_result.append(y_pred)
        write_log(card_id,tq,ta,rqa,y_pred)

    for key in data:
        print('{} : {} \n'.format(key, data[key]))
    
    
def open_audio(card_id):
    """
    get last element (recent previous result) from dict of that card_id and set
    the audio key (e.g. open audio or not)
    """
    result = data[card_id][-1]
    print('open or not {}'.format(result))
    if result == 0:
        print('open')
        return True
    else:
        return False

def read_ratio_from_estm2():
    """
    This function is for reading the trigger rate ratio from the previous 
    estimation task and set it as a trigger rate ratio.
    """
    previous_deck_id = int(deck_name.split('_')[0])-1
    previous_file_name = '0'+str(previous_deck_id)
    for file_path in glob.iglob('D:/Okayama U/2020-4 research/2020_09_Anki/model/log_'+previous_file_name+'*.txt'):
        print(file_path)
        size = os.path.getsize(file_path) 
        print(size)
        previous_log_file_name = file_path.split('\\')[-1].replace('.txt','')
        print(previous_log_file_name)
        if max(previous_log_file_name[20:])  and size != 0:
            print('can found')
            f = open(file_path,'r+')
            trigger_ratio = f.readlines()[-1].split(' ')[3].replace('\n','')
            print(trigger_ratio)
            set_rratio(trigger_ratio)

def read_ratio_from_estm():
    """
    This function is for reading the trigger rate ratio from the previous 
    estimation task and set it as a trigger rate ratio.
    """
    previous_deck_id = int(deck_name.split('_')[0])-1
    previous_file_name = '0'+str(previous_deck_id)
    file_list = glob.glob('D:/Okayama U/2020-4 research/2020_09_Anki/model/log_'+previous_file_name+'*.txt')
    #not_empty_file_path = [file_path for file_path in file_list if os.path.getsize(file_path) != 0] 
    recenttime_file_path = max([file_path.split('\\')[-1].replace('.txt','')[20:] for file_path in file_list if os.path.getsize(file_path) != 0])
    
    for file_estm in file_list:
        if recenttime_file_path in file_estm:
            f = open(file_estm,'r+')
            trigger_ratio = f.readlines()[-1].split(' ')[3].replace('\n','')
            print(trigger_ratio)
            set_rratio(trigger_ratio)           
            
def update_dict(y_pred):
    """
    update the predicted result into the temp dictionary.
    """
    data[card_id].append(y_pred)
    
def write_log(card_id,tq,ta,rqa,y_pred):
    """
    write the log file.
    """
    file.write("{} {} {} {} {}\n".format(card_id,tq,ta,rqa,y_pred))
    
def create_log(deck_id):
    """
    create the audio trigger log file.
    """
    global file
    file_name = 'D:/Okayama U/2020-4 research/2020_09_Anki/model/log_'+str(deck_name)+'_'+str(time.time())+'.txt'
    file = open(file_name,'w+')
    print(file)

def summary_trigger():
    """
    before closeing the log file, calculate the trigger rate and write into the
    last line of log file.
    """
    print(len(predict_result))
    if len(predict_result) != 0:
        count_zero = predict_result.count(0)
        count_one = predict_result.count(1)
        if count_zero == 0 and count_one == 0:
            trigger_rate = 0
        else: 
            trigger_rate = count_zero/(count_zero+count_one)
        file.write("count_auto_trigger {} {} {}\n".format(count_zero,count_one,trigger_rate))
            
def close_log():
    """
    Close the log file after finish learning task.
    """
    if file != '':
        if file.closed:
            print('already closed')
        else:
            summary_trigger()
            print('Can close')
            predict_result.clear()
            file.close()
