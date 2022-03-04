# Building an estimation model of learners' performance 

This repository contains the custom codes and tools develped to obtain the results reported in our article [1]:

**Data-preprocessing**

The function data_preparation.py contains the subroutines for computing the behavioral variables and carrying out the data the pre-processing steps on them. In that regard, it first computes the cdf of t_c, t_q, t_a and r_qa (See [2] for the details of variable definitions). Based on each of those, it defines a threshold value, which bounds the lower 95% of the relating data. We consider the data points as admissible, which have all their fields below relating thresholds. Namely, if all tc, tq, ta and rqa are below the threshold, we consider that data point as admissible. If any one of them is above the threshold, we discard that data point. 

The function then clones the data by copying only the admissible data points. In addition, while cloning, we remove the data points which have remember_or_forget 
as -1. That means the card is removed from the deck of other participants. So there is no point in considering that card or data point.

**Model development**

The function build_NN_model.py is used for building the estimator. We considered building a  model for every learning tasks. First, the input data file (pickle files) are loaded and the different behavioral variables stored in each column are separated.We drop the column which is not be used for build the model out. Next, StandardScaler is applied to the retained data. Subsequently, we reshape the list the train set and convert label data set to be categorical data which is required before fed into the model. then, we develop an artificial neural network which serves as a model for estimation of memory performance. We test the accuracy of the models and save them into the pickle files ready to be imported by the e-learning software.

**Integration of the model to the e-learning software**

The function add_on_model.py is integrated with the e-learning software for estimating in an online manner. First, the behavioral variables are calculated.

Should the operating mode of the e-learning software be 'Estimation', the variables are then prepared in the same way as described above. The pre-trained estimation model is loaded and the prepared variables are fed into it as inputs. Depending on the estimation result, the audio key (i.e. trigger or no-trigger) is set and stored in a log file.

Should the operating mode of the e-learning software be 'Full audio reinforcement',the audio key is set to 'ON' (alway triggering of audio). The audio trigger log file is updated accordingly.

**References**

[1] Investigation of the effects of audio-reinforcement on recollection rates of e-learning users,
P. Supitayakul, Z. Yücel, A. Monden, and P. Leelaprute,
Under review

[2] Identification of behavioral variables for efficient representation of difficulty in vocabulary learning systems,
P. Supitayakul, Z. Yücel, A. Monden, and P. Leelaprute,
International Journal of Learning Technologies and Learning Environments, vol. 3, no. 1, pp. 51–60, 2020.
