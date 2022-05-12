# Machine learning project 3

## SVM

The jupyter notebook SVM can be run to create the SVM, get the raw
signal data from the testing/training data sets. 
Then we train the SVM with the testing data and its correct
classifications. We can then predict the list of signal data 
from the testing set, and get a list of the 1200 classifications.

To run the code you can just run every cell, and it should end 
up creating a text file with the classifications. There is also a 
jupyter notebook file that converts the text file into the csv file 
to submit.

## Convolutional Neural Network

The file convoNeuralNet.py is the script for our neural network
you need the data for the project in a file labled MusicData
the scripts that make the spectrograms are testSpecGramScript.py and trainSpecGramScript2.py
for these to run there need to be two directories trainSpecGrams/ and testSpecGrams/
once the script is run those directories need to be moved into wrapper directories
the wrappers need to be labeled convoTrainData/ and convoTestData/
when all is done you should have two directories
dir1: ./convoTrainData/trainSpecGrams/[the spectrograms of the training data]
dir2: ./convoTestData/testSpecGrams/[the spectrograms of the testing data]
then you just run convoNeuralNet.py,
However this script DOES NOT work at the moment, the spectrograms are still 1440x1980
and therefore way to large to compute, more explanation in report.

## Fully Connected Neural Network
the jupyter notebook script MP3toMFCC.ipynb can be used to generate the needed npz's
make sure the files trainingMFCC.npz and testingMFCC.npz are in you working directory
also make sure that the music data is in a file labled MusicData
then you should be able to run fullConnectedNN.py
you may want to edit the epoch count on line 40 for a shorter runtime.
