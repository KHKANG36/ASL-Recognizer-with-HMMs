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

####1) Feature set organization (Data organization)
 - I implement five feature sets for ASL training and testing. (1) Ground : used the absolute difference between the left/right-hand x/y value and the nose x/y value, which serves as the "ground" value. (2) normalized Cartesian coordinates : use mean and standard deviation statistics and the standard score equation to account for speakers with different heights and arm length (3) polar coordinates : calculated polar coordinates with Cartesian to polar equations (4) delta difference : used the difference in values between one frame and the next frames as features (5) normalized polar coordinates : like feature set (2), used mean and standard deviation statistics in polar coordinates to account for speakers with different heights and arm length. 
 
####2) Model Selection 

The purpose of Model Selection is to tune the number of states for each word HMM prior to testing on unseen data. We should set the ideal number of HMMs states for best performance. For example, if we can express the ASL for "Chocolate" word with 4 states at best, each 4 states should be clearly separated with mean and variance of each state. Below is the visualization of mean and variance of 4 states for "Chocolate" word. (This is not the optimal but just one of the example.)

![Test image](https://github.com/KHKANG36/ASL-Recognizer-with-HMMs/blob/master/data/HMM_Chocolate.png)

In general, since we don't know exactly how many states would be ideal in HMMs, I explored three methods in this project to decide the optimal number of states for each word. Three models are as below:

1) Log likelihood using cross-validation folds (CV)
2) Bayesian Information Criterion (BIC)
3) Discriminative Information Criterion (DIC) 

1) Cross Validation: While DIC, BIC attempts to penalizing the model complexity, cross validation should be exactly the selector socre on observed data.Therfore, if test data sets are pretty much similar to train data sets, it will provide the most accurate result. However, since it requires running through the data multiple times, over each folds, the expense will reult in a very slow runs. Therefore, it is not suitable with large datasets. CV technique handle the overfitting problem by training on as many folds as are passed in as its hyperparameter. 2) BIC: This model penalizes model complexity. The main advantage of BIC is that it provides simpler models by penalizing models that have high complexity with the penalize parament 'p'. 3) DIC: DIC is calculated by subtracting Loglikelihood of given word and average Loglikelihood for other words data. DIC is rather more complex model than BIC in terms of penalizing, because it calculate the Loglikelidhood of all the words and compare them with given word. 

####3) Recognizer 
Using the five feature sets created and the three model selectors (15 possible combinations) with HMMs library, I trained all words in datasets and tested the words in test datasets. For the accuracy metrics, I used WER (Word Error Rate). The result is as below:

1. Total Result (WER):

|      |CV    |BIC      |DIC      |
|:---: |:---: |:------: |:-------:|
|Ground|59.6% |__55.0%__|57.3%    |
|Norm  |66.3% |61.2%    |59.6%    |
|Polar |61.2% |__54.5%__|__54.4%__|
|Delta |61.2% |61.8%    |62.9%    |
|Custom|63.5% |59.6%    |57.3%    |

2. The performance analysis: 
As we discussed in section 2, the fastest model to test on was also BIC model. In addition, the overall accuracy of BIC was also the best among three models. In terms of features, polar shows the best accuracy result. Top 3 performance combination were (1) DIC with Polar features (54.4% WER), (2) BIC with Polar features (54.5% WER) and (3) BIC with Ground features (55.0% WER). First of all, polar coordinate features was able to more efficiently differenciate words than ground coordinate features. I think that using angle and distance might be differenciate the hand location than x,y coordinate because the variation between angle and distance is larger than that of x,y coordinate. Therefore, the "nomalized polar" could not help to reduce the WER because normalization might reduce the variation among data. BIC might prevent the overfitting effectively by penalizing the model complexity, and it might show the best accuracy in test sets. 

## Future Works
Obviously, like NLP processing, understanding the relationship between words in sentence is the most important in order to improve the accruacy. Therefore, we can enhance the performance with SLM(Statictical Language Model) data using 1,2,3-gram statistics. The basic idea for this is that each word has some probability of occurrence within the set, and some probability that it is adjacent to specific other words. I will try this technique, and compare the result with this project soon. 
