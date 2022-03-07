# ANN-based audio reinforcement for computer assisted rote learning

This repository contains the custom codes and tools developed to obtain the results reported in our article [1]:
Note that the raw data set contains privacy sensitive information, which is concealed in this release.

**Activity log files**

The activity log files are recorded from the e-learning software Anki [2]. They contain three basic kinds of information as temporal, identifier and evaluation.

The temporal variables are registered in  UNIX time at millisecond resolution and include tp, tf, and te. Here, tp denotes the time of prompt, i.e. the instant when the Q-face of a card appears. In addition,  tf represents the time of flip, i.e. the instant when the learner presses the ''Show Answer'' button and discloses the A-face of the card. Finally,  te stands for the time of evaluation. Namely, it is the instant when the learner assesses the difficulty of a card by choosing one of  ''Again'', ''Good'' or ''Easy'', which is also registered as the evaluation variable. For more detail, please see https://docs.ankiweb.net/#/ .

On the other hand, the identifier variables are integer codes used to determine the deck or card that is being studied (i.e.displayed) at a given time instant (e.g. deck ID, card ID). Note that each log file is associated with a single user. Namely, the software recorded one log file into the account of each user. In addition, each line of the activity log file  corresponds to a single action of the user which is considered as a reaction to the software. The structure of each line of data is as follows:  [UNIX time], function name, data in detail (i.e. flags, queue).

**Memory score file**

This file stores the performance score of the participants in the prior (P), short-term (S) and mid-term (M) memory tests. The file contains 6 sheets, 3 for the exploration stage and 3 for the verification stage. Concerning each stage, the scores concerning P, S and M tests are stored in separate sheets.

Regarding the coding of memory test scores, if the participant recalls the answer of a certain query successfully, we register the score of that card with 1, otherwise 0. Different columns contain the scores of different participants and different rows contain the information concerning different cards. In addition, the card ID, Q-face and A-face information is also written on the left hand side of each row as a key for matching with the activity log files.

**Model development**

The function NN_model_builder.py is used for building the estimator. We considered building a model for every learning task. First, the input data file (compressed as pickle files) are loaded and the different behavioral variables stored in each column are separated. We drop the column, which is not used for building the model. Next, StandardScaler is applied to the retained data. Subsequently, we reshape the train set and convert the labels to categorical data before feeding them into the model. Finally, we develop an artificial neural network which serves as a model for estimation of memory performance as improvement or deterioration. We test the accuracy of the model and save them into the pickle files ready to be imported by the e-learning software.

**Integration of the model to the e-learning software**

The function NN_model_integration.py is integrated with the e-learning software for estimating memory performance in an on-the-fly manner. First, the behavioral variables are calculated.

Should the operating mode of the e-learning software be 'Estimation', the variables are then prepared in the same way as described above. The pre-trained estimation model is loaded and the prepared variables are fed into it as inputs. Depending on the estimation result, the audio key (i.e. trigger or no-trigger) is set and stored in a log file.

Should the operating mode of the e-learning software be 'Full audio reinforcement',the audio key is set to 'ON' (constant triggering of audio). The audio trigger log file is updated accordingly.

**References**

[1] ANN-based audio reinforcement for computer assisted rote learning,
Authors, Under review

[2] D. Elmes, “Anki - friendly, intelligent flashcards.” https://ankiweb.net/about, 2021.

[3] P. Supitayakul, Z. Yücel, A. Monden, and P. Leelaprute,
Identification of behavioral variables for efficient representation of difficulty in vocabulary learning systems,
International Journal of Learning Technologies and Learning Environments, vol. 3, no. 1, pp. 51–60, 2020.

[4] P. Supitayakul, Displaying visual stimuli and recording audio, https://github.com/Parisa-S/Displaying-visual-stimuli-and-recording-audio.

