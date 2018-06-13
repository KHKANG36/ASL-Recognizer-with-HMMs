# Sign Language Recognition System

In this project, I implemented a word recognizer for American Sign Language video sequences, demonstrating the power of probabalistic models. In particular, I employs hidden Markov models (HMM's) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the RWTH-BOSTON-104 Database). In this video, the right-hand x and y locations are plotted as the speaker signs the sentence. 
![Test image](https://github.com/KHKANG36/ASL-Recognizer-with-HMMs/blob/master/data/ASL_DB.png)
I orgaznied a variety of feature sets for training and testing, as well as implemented three different model selection criterion to determine the optimal number of hidden states for each word model. Finally, I implemented the recognizer and compare the effects the different combinations of feature sets and model selection criteria.

## Requirements

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [jupyter](http://ipython.org/notebook.html)
- [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)

Notes: 
1. It is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python and load the environment included in the "Your conda env for AI ND" lesson.
2. The most recent development version of hmmlearn, 0.2.1, contains a bugfix related to the log function, which is used in this project.  In order to install this version of hmmearn, install it directly from its repo with the following command from within your activated Anaconda environment:
```sh
pip install git+https://github.com/hmmlearn/hmmlearn.git
```

## Code

All running code for this project is implemented in 'asl_recognizer.ipynb' and all required data are uploaded in this project repo. Three different model is implemented in 'my_model_selectors.py', and model recognizer for training and testing is written at 'my_recognizer.py'

## Run

In a terminal or command window, run the following command:

`jupyter notebook asl_recognizer.ipynb`

This will open the Jupyter Notebook software and notebook in your browser which is where you will directly edit and run your code. 

## Project Description 

1) Data 

