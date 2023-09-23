# Project1-LR-NB-TxtClassification
In this project, you will implement and evaluate Naive Bayes and Logistic Regression for text
classification.

Download the spam/ham (ham is not spam) datasets (see the zip file). The datasets were
used in the Metsis et al. paper [1]. There are three datasets. Each data set is divided into two
(sub)sets: training set and test set. Each of them has two directories: spam and ham. All
files in the spam and ham folders are spam and ham messages respectively.

Convert the text data into a matrix of features × examples (namely our canonical data
representation), using the following approaches.: Bag of Words model and Bernoulli model.
Thus you will convert each of the three “text” datasets into two datasets, one using the Bag
of Words model and the other using the Bernoulli model. You can use any text processing tool or library (e.g., NLP toolkit) to accomplish this task.

Implement the multinomial Naive Bayes algorithm for text classification. 
Note that the algorithm uses add-one laplace smoothing. Make sure that you do all the calculations in log scale to avoid underflow. Use your algorithm to learn from the training set and report accuracy on the test set. Important: Use the datasets generated using the Bag of words model and not the Bernoulli model for this part.

Implement the discrete Naive Bayes algorithm we discussed in class. To prevent zeros, use add-one Laplace smoothing. Important: Use the datasets generated using the Bernoulli model and not the Bag of
words model for this part.

Implement the MCAP Logistic Regression algorithm with L2 regularization that we discussed in class (see Mitchell’s new book chapter). Learn parameters using the 70% split, treat the 30% data as validation data, and use it to select a value for λ. Then, use the chosen value of λ to learn the parameters using the full training set and report accuracy on the test set. Use gradient ascent for learning the weights (you have to set the learning rate appropriately.
Otherwise, your algorithm may diverge or take a long time to converge). Do not run gradient ascent until convergence; you should put a suitable hard limit on the number of iterations.

Run the SGDClassifier from scikit-learn on the datasets. Tune the parameters (e.g., loss function, penalty, etc.) of the SGDClassifier using GridSearchCV in scikit-learn. Use the datasets generated using both the Bernoulli model and the Bag of
words model for this part.
