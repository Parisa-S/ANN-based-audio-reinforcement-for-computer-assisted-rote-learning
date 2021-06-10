# Building an estimation model of learners performance 

This repository contains the custom codes and tools develped to obtain the results reported in our article [1]:

**Data-preprocessing**

The function data_preparation.py contains the subroutines for computing the behavioral variables and carrying out the data the pre-processing steps on them. In that regard, it first computes the cdf of t_c, t_q, t_a and r_qa (See [2] for the details of variable definitions). Based on each of those, it defines a threshold value, which bounds the lower 95% of the relating data. We consider the data points as admissible, which have all their fields below relating thresholds. Namely, if all tc, tq, ta and rqa are below the threshold, we consider that data point as admissible. If any one of them is above the threshold, we discard that data point. 

The function then clones the data by copying only the admissible data points. In addition, while cloning, we remove the data points which have remember_or_forget 
as -1. That means the card is removed from the deck of other participants. So there is no point in considering that card or data point.

**Model development**

The function estimator_model.py is used for building the estimator. We considered building a devoted model for each of the three types of learning task ( i.e. relatively easy verbal task, relatively hard verbal task, and numerical task). First, the pre-processed data file (pickle files) are loaded and the different behavioral variables stored in each column are separated. Next, log-normalization is applied followed by principal component analysis (PCA) to the retained data. Subsequently, in order to  increase the number of data points, we run synthetic minority oversampling (SMOTE), which completes data processing. then, we develop a Naive Bayes classifier which serves as a model for estimation of likelihood of remembering/forgetting. We test the accuracy of the models and save them into  the pickle files ready to be imported by the e-learning software.

**Integration of the model to the e-learning software**

The function controller.py is integrated with the e-learning software for estimating in an online manner the learners' likelihood of remembering/forgetting. First, the behavioral variables are calculated. 

Should the operating mode of the e-learning software be 'Estimation', the variables are then pre-processed in the same way as described above. The pre-trained estimation model corresponding to the type of the learning material that is being studied is loaded and the pre-processed variables are fed into it as inputs. Depending on the estimation result, the audio key (i.e. trigger or no-trigger) is set and stored in a log file.
    
Should the operating mode of the e-learning software be 'Random', an arbitrary number between 0 and 1 is drawn and compared to a threshold, which is adjusted as the empirical rate of forgetting. If it is lower than the threshold, then the audio key is set to be 'ON' (i.e. audio will be triggered). Finally,  the audio trigger log file is updated 

Should the operating mode of the e-learning software be 'Visual',  the leaner is assumed to remember any card in any deck and the audio key is set to 'OFF' (no triggering of audio). The audio trigger log file is updated accordingly.
    

**References**

[1] Investigation of the effects of audio-reinforcement on recollection rates of e-learning users,
P. Supitayakul, Z. Yücel, A. Monden, and P. Leelaprute,
Under review

[2] Identification of behavioral variables for efficient representation of difficulty in vocabulary learning systems,
P. Supitayakul, Z. Yücel, A. Monden, and P. Leelaprute,
International Journal of Learning Technologies and Learning Environments, vol. 3, no. 1, pp. 51–60, 2020.
