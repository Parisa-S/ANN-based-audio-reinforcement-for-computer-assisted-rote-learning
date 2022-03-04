# -*- coding: utf-8 -*-

import pickle
import numpy as np
import time
import os
from datetime import datetime


# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr

start_time = ''
flip_time = ''
ans_time = ''
evalute_bt = ''
model = ''
card_id = ''
data = dict()
file = ''
modal = ''
difficulty = ''
r_ratio = 0
predicted_result_tmp = ''
predict_results_bin = []
predict_results_prob = []
    
def set_starttime(time):
    global start_time
    start_time = time
    print('start_time: {}'.format(start_time))
    
def set_fliptime(time):
    global flip_time
    flip_time = time
    print('flip_time: {}'.format(flip_time))
    
def set_anstime(time):
    global ans_time
    ans_time = time
    print('ans_time: {}'.format(ans_time))

def set_evalute_bt(eval_bt):
    global evalute_bt
    evalute_bt = eval_bt
    print('evalute_bt: {}'.format(eval_bt))
    
def set_cardID(cid):
    global card_id
    card_id = cid
    print('card_id: {}'.format(card_id))
    
def set_deckID(did):
    global deck_id
    deck_id = did
    print('deck_id: {}'.format(deck_id))
    
def set_deckinfo(cids,d_name):
    global deck_name
    deck_name = d_name
    print('deckinfo: {}'.format(deck_name))

def set_profile_name(name):
    global profile_name
    profile_name = name
    print('profile_name: {}'.format(profile_name))
    create_directory(profile_name)

def create_directory(name):
    located_path = '../log/'
    path_name = name
    
    try:
        os.mkdir(located_path+path_name)
    except OSError:
        print("create_directory() : Failed to create new folder")


def set_rratio(ratio):
    global r_ratio
    r_ratio = ratio
    print('r_ratio: {}'.format(r_ratio)) 
    
def deck_type(deck_name):
    #name_format : deckdiff_modal
    global modal,difficulty
    modal = deck_name[8]
    difficulty = deck_name[3].lower()
    print('deck_type: modal {}, difficulty {}'.format(modal,difficulty))
    return difficulty,modal

"""
after user selected the deck they want to learn 
anki will call load_model when open the deck
if the modal of that deck is ANN estimation,then load 2 models
that is model for standardScaler and ANN model.
"""
def load_model(deck): 
    deck_name,modal = deck_type(deck) 

    if modal == 'E':
        model_filename = 'model/nn_model.pkl'
        scaler_filename = 'model/normalize_scaler.pkl'
        with open(model_filename,'rb') as model_file:
            global model_nn
            model_nn = pickle.load(model_file)
        with open(scaler_filename,'rb') as pre_model_file:
            global model_scaler
            model_scaler = pickle.load(pre_model_file)
    else:
        print('load_model(deck): Error. Can not load model')

"""
This function is used to prepare data which is required as input in ANN model
"""
def prepared_difficulty_data(difficulty):
    easy,hard,numerical = -1,-1,-1
    
    if difficulty.lower() == 'e':
        easy,hard,numerical = 1,0,0
    elif difficulty.lower() == 'h':
        easy,hard,numerical = 0,1,0
    elif difficulty.lower() == 'n':
        easy,hard,numerical = 0,0,1
    else:
        print('prepared_difficulty_data(difficulty) : Wrong difficulty types!'+ difficulty)
    return easy,hard,numerical

"""
This function is used to prepare data which is required as input in ANN model
we decided to used human audio for numeric deck and bell sound for verbal deck
"""
def prepared_modal_data(difficulty):
    visual, audio, bell = -1,-1,-1
    
    if difficulty.lower() == 'e' or difficulty.lower() == 'h':
        visual, audio, bell = 0,0,1
    elif difficulty.lower() == 'n':
        visual, audio, bell = 0,1,0
    else:
        print('prepared_modal_data(difficulty) : Wrong modal types!'+ difficulty)
    return visual, audio, bell

"""
This function is used to prepare data which is required as input in ANN model
In case that predicted result is 0 (not play) then the audio attachment will 
set to be 0. 
"""
def check_sound_attachment(difficulty,predicted_result):
    audio_attachment,bell_attachment = -1,-1
    
    if difficulty.lower() == 'e'or difficulty.lower() == 'h':
        if predicted_result == 0:
            audio_attachment,bell_attachment = 0,0
        else:
            audio_attachment,bell_attachment = 0,1
    elif difficulty.lower() == 'n' :
        if predicted_result == 0:
            audio_attachment,bell_attachment = 0,0
        else: 
            audio_attachment,bell_attachment = 1,0    
    else:
        print('check_sound_attachment() : check error')
    return audio_attachment,bell_attachment


"""
predict() function is used for estimate memory performance of the user.
After user evaluated difficulty and flush the card they learned.
Anki call predict() function immidiately.

First the function will check the number of viewing of this card.
In case that user learn this card for the first time,
there is no data for the audio_attachment_prev,bell_attachment_prev,
tq_prev, ta_prev, eval_prev. So, we can not prepare data and predict the result.
    
However, after the first time, we can have the data of previous one.
So, we can prepare the data and predict it by pre-train model.

If the modal of selected deck is ANN estimation, Anki will try to predict.
But if the modal of selected deck is full audio reinforcement then Anki set
the result to be 1(play).

After the estimation process is finished. 
Anki update the dictionary and estimation log file.

"""
def predict():
    global new_feats
    
    n_current_viewing = len(data[card_id]['eval'])

    if n_current_viewing == 0:
        tq_current = int(float(flip_time*1000))-int(float(start_time*1000))
        ta_current = int(float(ans_time*1000))- int(float(flip_time*1000))
        eval_current = evalute_bt
        predicted_result = predicted_result_tmp
        y_pred_prob = [0,0]

    else:
        #prepare the data into the required shape
        easy, hard, numerical = prepared_difficulty_data(difficulty)
        visual, audio, bell = prepared_modal_data(difficulty)
        audio_attachment_prev = data[card_id]['audio_attachment'][-1]
        bell_attachment_prev = data[card_id]['bell_attachment'][-1]
        tq_current = int(float(flip_time*1000))-int(float(start_time*1000))
        ta_current = int(float(ans_time*1000))- int(float(flip_time*1000))
        eval_current = evalute_bt
        tq_prev =  data[card_id]['tq'][-1]
        ta_prev =  data[card_id]['ta'][-1]
        eval_prev =  data[card_id]['eval'][-1]
        
        temp = [easy, hard, numerical,\
            visual, audio, bell,\
            n_current_viewing,\
            audio_attachment_prev, bell_attachment_prev,\
            tq_current, ta_current, eval_current,\
            tq_prev, ta_prev, eval_prev ]
    
        x_test = temp.copy()
        
        if modal == 'E':

            z = np.array(x_test).reshape(-1, len(temp)) #reshape each data point from 3D to 2D
            x_test_norm = model_scaler.transform(z) #scale the input data 
            x = x_test_norm.reshape(-1,1, len(temp))  #reshape it back to 3D

            y_pred_prob = model_nn.predict_on_batch(x) # predict on the new input / return the prob of binary result
            y_pred_binary = np.round_(y_pred_prob) # convert the prob to binary result
                     
            predict_results_prob.append(y_pred_prob)  # save predictions into list for prob
            predict_results_bin.append(y_pred_binary)  # save predictions into list for binary
            online_predict = np.array(y_pred_binary).reshape(-1,2)
            predicted_result = np.argmax(online_predict, axis=1)
        
            model_nn.train_on_batch(x, y_pred_binary)  # runs a single gradient update 
                                  
        else:
            print('predict(): using All Audio')
            predicted_result = 1
            y_pred_prob = [0,0]
            
        
    audio_attachment,bell_attachment = check_sound_attachment(difficulty,predicted_result)       
    
    update_dict(audio_attachment,bell_attachment,tq_current,ta_current,eval_current,int(predicted_result))
    write_log(card_id,tq_current,ta_current,y_pred_prob,predicted_result)
 
    
"""
Before the next card appear, Anki check the audio condition by get 
the last element from the dict of that card (lastest estimation result). 
If that card is called for the first time, then Anki random the number 
to decided whether to play or not play.
"""
def get_result_audio(card_id):
    global predicted_result_tmp
    #get last element from dict of that card
    n_current_viewing = len(data[card_id]['eval'])
    if n_current_viewing == 0:
        if modal == 'E':
            predicted_result_rand= np.random.random()
            predicted_result_tmp = np.round_(predicted_result_rand)   
        else:
            predicted_result_tmp = 1
        result = predicted_result_tmp
    else:
        result = data[card_id]['pred_result'][-1]
        
    return result

"""
open_audio(card_id) is used to set the switch of turn on / off audio in Anki.
"""   
def open_audio(card_id):
    
    result = get_result_audio(card_id)   
    if result == 1:
        print('open audio() : open')
        return True
    else:
        print('open audio() : off')
        return False
    
"""
update the new data to dictionary
"""
def update_dict(audio_attachment,bell_attachment,tq_current,ta_current,eval_current,y_pred):
    data[card_id]['audio_attachment'].append(audio_attachment)
    data[card_id]['bell_attachment'].append(bell_attachment)
    data[card_id]['tq'].append(tq_current)
    data[card_id]['ta'].append(ta_current)
    data[card_id]['eval'].append(eval_current)
    data[card_id]['pred_result'].append(y_pred)

"""
write the estimation log file
"""    
def write_log(card_id,tq,ta,y_pred_prob,y_pred):
    file.write("{} {} {} {} {}\n".format(card_id,tq,ta,y_pred_prob,y_pred))

"""
create the new log file when start the new learning session
"""    
def create_log(deck_id):
    global file
    file_name = '../log_'+str(deck_name)+'_'+str(time.time())+'.txt'
    file = open(file_name,'w+')
    print('create_log() : {}'.format(file))
    

"""
close the log file after finish writing.
"""            
def close_log():
    if file != '':
        if file.closed:
            filename = '../pkllog_'+str(deck_name)+'_'+str(time.time())+'.pkl'
            pickle.dump( data, open(filename , "wb" ) )
            print('close_log() : already closed')
        else:
            #summary_trigger()
            print('close_log() : Can close')
            filename = '../pkllog_'+str(deck_name)+'_'+str(time.time())+'.pkl'
            pickle.dump( data, open(filename , "wb" ) )
            data.clear()
            file.close()
