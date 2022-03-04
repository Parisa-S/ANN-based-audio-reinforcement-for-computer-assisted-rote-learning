# Building an estimation model of learners' performance 

This repository contains the custom codes and tools develped to obtain the results reported in our article [1]:
Note that the raw data set contains privacy sensitive information, which is concealed in this release.

**Activity log files**

The activity log files are recorded from the e-learning software Anki [2]. They contain two basic kinds of information as (i) temporal and (ii) identifier.

The temporal variables are registered in  UNIX time at millisecond resolution and include tp, tf, and te. Here, tp denotes the time of prompt, i.e. the instant when the Q-face of a card appears. In addition,  tf represents the time of flip, i.e. the instant when the learner presses the ''Show Answer'' button and discloses the A-face of the card. Finally,  te stands for the time of evaluation, i.e. the instant when the learner assesses the difficulty of a card by choosing one of  ''Again'', ''Good'' or ''Easy''. For more detail, please see https://docs.ankiweb.net/#/ .

On the other hand, the identifier variables are integer codes used to determine the deck or card that is being studied (i.e.displayed) at a given time instant (e.g. deck ID, card ID). Note that each log file is associated with a single user. Namely, the software recorded one log file into the account of each user. In addition, each line of the activity log file  corresponds to a single action of the user which is considered as a reaction to the software. The structure of each line of data is as follows:  [unix time], function name, data in detail (i.e. flags, queue).

**Model development**

The function build_NN_model.py is used for building the estimator. We considered building a  model for every learning tasks. First, the input data file (pickle files) are loaded and the different behavioral variables stored in each column are separated.We drop the column which is not be used for build the model out. Next, StandardScaler is applied to the retained data. Subsequently, we reshape the list the train set and convert label data set to be categorical data which is required before fed into the model. then, we develop an artificial neural network which serves as a model for estimation of memory performance. We test the accuracy of the models and save them into the pickle files ready to be imported by the e-learning software.

**Integration of the model to the e-learning software**

The function add_on_model.py is integrated with the e-learning software for estimating in an online manner. First, the behavioral variables are calculated.

Should the operating mode of the e-learning software be 'Estimation', the variables are then prepared in the same way as described above. The pre-trained estimation model is loaded and the prepared variables are fed into it as inputs. Depending on the estimation result, the audio key (i.e. trigger or no-trigger) is set and stored in a log file.

Should the operating mode of the e-learning software be 'Full audio reinforcement',the audio key is set to 'ON' (alway triggering of audio). The audio trigger log file is updated accordingly.

**References**

[1] ANN-based audio reinforcement for computer assisted rote learning,
P. Supitayakul, Z. Yücel, A. Monden, and P. Leelaprute,
Under review

[2] D. Elmes, “Anki - friendly, intelligent flashcards.” https://ankiweb.net/about, 2021.

[3] Identification of behavioral variables for efficient representation of difficulty in vocabulary learning systems,
P. Supitayakul, Z. Yücel, A. Monden, and P. Leelaprute,
International Journal of Learning Technologies and Learning Environments, vol. 3, no. 1, pp. 51–60, 2020.

[4] P. Supitayakul, Displaying visual stimuli and recording audio, https://github.com/Parisa-S/Displaying-visual-stimuli-and-recording-audio.

