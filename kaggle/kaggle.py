
# coding: utf-8

# In[1]:

import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV # use Grid Search to select best XGB model using CV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit # for cross validation
    # Full guide: http://scikit-learn.org/stable/modules/cross_validation.html
    # train_test_split creates one split based off of test size specified
    # KFold divides all the samples in k folds of samples, similiar to train_test_split except you get all the folds
    # StratifiedKFold tries to ensure classes are balanced in each fold
    # RepeatedKFold/RepeatedStratifiedKFold repeats K-Fold n times with different randomization in each repetition
    # ShuffleSplit shuffles the data before splitting, like train_test_split with shuffle=True, but does this for multiple iterations
    # StratifiedShuffleSplit is like ShuffleSplit, but ensures class balance like StratifiedKFold
    # cross_val_score allows for computation of the score directly from model and data
from scipy.stats import multivariate_normal

# XGBoost
# Installation: https://github.com/dmlc/xgboost/tree/master/python-package, read instructions
#   MacOS:
#   brew install gcc@5
#   pip install xgboost
# Usage: http://xgboost.readthedocs.io/en/latest/python/python_intro.html#install-xgboost
# Parameters: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
#   https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# Dataset: http://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html#preparation-of-the-dataset
# Theory: http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/
# Example with GridSearchCV: https://www.kaggle.com/phunter/xgboost-with-gridsearchcv
# Issue with multiclass: https://stackoverflow.com/questions/44070118/valueerror-multiclass-format-is-not-supported-xgboost

# Flag to indicate old data or new data
# TAKE THIS PART OUT WHEN SUBMITTING
old_or_new = 'new'

if old_or_new == 'old':
    data_folder = 'old_data/'
elif old_or_new == 'new':
    data_folder = 'new_data/'

### ~~~~~ Data Management ~~~~~ ###

# Prepare training and testing data
# old_or_new specifies if data is old data set or new data set
def getData():    
        
    # Names of training and testing CSV files, must be in same directory
    train_file_name = 'train.csv' 
    test_file_name = 'test.csv'
    
    # Load CSV files in numpy arrays
    #train_file = open(train_file_name)
    #test_file = open(test_file_name)
    train_data_matrix = np.loadtxt(data_folder + train_file_name,dtype=int,delimiter=',')
    test_data_matrix = np.loadtxt(data_folder + test_file_name,dtype=int,delimiter=',')
    
    # Split data into testing and training matrices
    # Each row is an observation, each column is a feature/label
    X_train = train_data_matrix[:,1:]
    Y_train = train_data_matrix[:,0] - 1 # subtract 1 because XGBoost accepts class values from [0,num_class]
    X_test = test_data_matrix
    
    # Normalize data
    #X_max = np.amax(np.stack((np.amax(X_train,axis=0),np.amax(X_test,axis=0))),axis=0)
    #X_train = X_train
    
    return X_train, Y_train, X_test

(X_train, Y_train, X_test) = getData()
print X_train, X_train.shape, type(X_train)
print Y_train, Y_train.shape, type(Y_train)
print X_test, X_test.shape, type(X_test)


# CURRENTLY UNUSED
# Split data up for K-fold cross validation
# Specify number of folds K, supply training data
def crossValidationSplit(K, X_train, Y_train):
    
    if K <= 1:
        print "Need to specify K > 1"
        return
    
    # First get number of observations
    N = X_train.shape[0]
    
    # Try to divide observations into evenly spaced buckets, last bucket will contain remainder
    bucket_size = int(np.floor(N/K)) # shouldn't need to floor if integer division, but just in case
        # cast to int to avoid indexing issues/warnings
    
    # Distribute data into lists of size K
    X_train_folds = [0 for k in range(K)]
    Y_train_folds = [0 for k in range(K)]
    for k in range(K):
        # Every bucket besides the last one should have bucket_size observations
        if (k+1) != K:
            X_train_folds[k] = X_train[k*bucket_size:(k+1)*bucket_size,:]
            Y_train_folds[k] = Y_train[k*bucket_size:(k+1)*bucket_size]
        # Last bucket contains remainder of elements that weren't evenly distribbuted
        else:
            X_train_folds[k] = X_train[k*bucket_size:,:]
            Y_train_folds[k] = Y_train[k*bucket_size:]
        
    return X_train_folds, Y_train_folds

K = 6
(X_train_folds, Y_train_folds) = crossValidationSplit(K,X_train,Y_train)
#for k in range(K):
#    print('Bucket {}: X: {}, Y: {}'.format(k+1, X_train_folds[k].shape, Y_train_folds[k].shape))


# In[2]:

### ~~~~~ XGBoost (Training and Testing) ~~~~~ ###

# Load in "fake" labels
sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]-1

# Load data from np matrix into XGBoost
# Can alternative load data from files directly
dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_sample) # label doesn't mean anything for dataset we're trying to predict

# Set parameters
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
param = {}
# Fixed
param['booster'] = 'gbtree' #tree-based model
param['objective'] = 'multi:softmax' #indicates multiclass classification, returns classes, not probabilities
param['num_class'] = 10 #need to specify number of classes when using multi:softmax
param['silent'] = 1 #1 to suppress messages
param['eval_metric'] = 'merror' #evaluation metric is classification error
# Tunable
param['eta'] = 0.5 #learning rate
param['max_depth'] = 10 #maximum depth of a tree
param['min_child_weight'] = 11 #tree-based parameters
param['gamma'] = 0.2
param['subsample'] = 0.8
param['colsample_bytree'] = 0.7
param['lambda'] = 1 #L2 regularization, Ridge
param['alpha'] = 0 #L1 regularization, Lasso
#param['nthread'] = 5 #specify number of cores used in parallel processing, leave it off because it breaks things
# Convert dictionary to list
plst = param.items()

# Evaluation set
eval_set=[(dtrain,'train'),(dtest,'test')]

# Train model
num_round = 230# i.e. number of estimators, make low for fast algo
bst = xgb.train(plst, dtrain, num_round, eval_set, verbose_eval=True)

# Predict using model
ypred_XGB = bst.predict(dtest) #on test set
ypred_on_train_XGB = bst.predict(dtrain) #on training set

# Accuracy results:
# 67.5% 1000 rounds, 10 depth, 11 min_child_weight, 0.8 subsample, 0.7 colsample_bytree, 1 lambda, 0 alpha
# 68.5%: 180-200 rounds, 6+ depth, same else from 67.5%
# 69%: gamma 0.2, 200/250 rounds, same else from 68.5% (68% for gamma 0.4) (with 180,300,500,1000 rounds, still 68%-68.5%)
# 69.5%: 225 rounds, same else
# 70%: 220,229-233 rounds, same else

# Note: Can't get cross-val score here using sklearn, need to use XGBClassifier
 
#print bst.eval(dtrain)
#print bst.eval(dtest)


# In[3]:

### ~~~~~ XGBoost (Accuracy Statistics) ~~~~~ ###

print 'XGBoost:\n'

train_error_XGB = np.sum(np.equal(ypred_on_train_XGB,Y_train))/float(ypred_on_train_XGB.size)
print 'Training percent correct: ', train_error_XGB

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_XGB = np.sum(np.equal(ypred_XGB+1,Y_sample))/float(ypred_XGB.size)
print 'Testing percent correct: ', test_error_XGB

print Y_sample.astype(int)
print ypred_XGB.astype(int)+1


# In[4]:

### ~~~~~ XGBoost on training set split into training and testing (Cross-Validation) ~~~~~ ###

folds = 5

# Why do I get the same accuracies for multiple iterations despite this being a random split?
# Because it's the same random split every time due to a fixed seed
#X_train_split,X_test_split,Y_train_split,Y_test_split = train_test_split(X_train, Y_train, 
#                                                                         test_size=float(1)/folds, #inverse of number of folds
#                                                                         random_state=42,
#                                                                        )

# Splitting strategy
#kf = KFold(n_splits=folds) # won't work because the class labels are ordered in training set
kf = StratifiedKFold(n_splits=folds)

i = 0
scores_XGB_CV = []
for train_index, test_index in kf.split(X_train,Y_train):
        
    # Split data
    X_train_split, X_test_split = X_train[train_index], X_train[test_index]
    Y_train_split, Y_test_split = Y_train[train_index], Y_train[test_index]

    # Load data from np matrix into XGBoost
    # Can alternative load data from files directly
    dtrain = xgb.DMatrix(X_train_split, label=Y_train_split)
    dtest = xgb.DMatrix(X_test_split, label=Y_test_split)

    # Set parameters
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    #param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax'}
    param = {}
    # Fixed
    param['booster'] = 'gbtree' #tree-based model
    param['objective'] = 'multi:softmax' #indicates multiclass classification, returns classes, not probabilities
    param['num_class'] = 10 #need to specify number of classes when using multi:softmax
    param['silent'] = 1 #1 to suppress messages
    param['eval_metric'] = 'merror' #evaluation metric is classification error
    # Tunable
    param['eta'] = 0.5 #learning rate
    param['max_depth'] = 10 #maximum depth of a tree
    param['min_child_weight'] = 11 #tree-based parameters
    param['gamma'] = 0.2
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.7
    param['lambda'] = 1 #L2 regularization, Ridge
    param['alpha'] = 0 #L1 regularization, Lasso
    #param['nthread'] = 4 #specify number of cores used in parallel processing
    # Convert dictionary to list
    plst = param.items()

    # Train model
    num_round = 230 # i.e. number of estimators, make low for fast algo
    bst = xgb.train(plst, dtrain, num_round)

    # Predict using model
    ypred_XGB_CV = bst.predict(dtest) #on test set
    ypred_on_train_XGB_CV = bst.predict(dtrain) #on training set

    # Print error for splt predicting testing and training sets
    # Error is equivalent to score in GridSearch? Makes sense then, since GridSearch uses CV...
    print ('Fold %d:' % (i+1))
    train_error_XGB_CV = np.sum(np.equal(ypred_on_train_XGB_CV,Y_train_split))/float(ypred_on_train_XGB_CV.size)
    print 'Training percent correct: ', train_error_XGB_CV
    test_error_XGB_CV = np.sum(np.equal(ypred_XGB_CV,Y_test_split))/float(ypred_XGB_CV.size)
    scores_XGB_CV.append(test_error_XGB_CV)
    print 'Testing percent correct (cross-val score): ', test_error_XGB_CV
    i += 1

print '\nCross-validation score (mean of all folds): ', np.mean(scores_XGB_CV)
# Compare this with the results from grid search


# In[5]:

### ~~~~~ XGBoost with Grid Search (Training and Testing) ~~~~~ ###

# Classifier to perform grid search for is XGB
xgb_model = xgb.XGBClassifier()

# List all parameters
#print xgb_model.get_params().keys()

"""
# Fine grid search
parameters = {#'nthread':[4], #when use hyperthread, xgboost may become slower, let this be 1 for now
              'objective': ['multi:softmax'], #indicates multiclass classification
              'learning_rate': np.linspace(0.1,1,10), #learning rate, aka 'eta' value, check 0.1 to 1
              'max_depth': np.linspace(3,10,8).astype(int), #maximum depth of a tree, check 3 to 10
              'min_child_weight': np.linspace(1,15,15), #tree parameters
              'gamma': np.linspace(0,10,11),
              'subsample': np.linspace(0.5,1,6),
              'colsample_bytree': np.linspace(0.5,1,6),
              'reg_lambda': [1,10,50,100,500,1000],
              'reg_alpha': np.linspace(0,10,11),
              'n_estimators': [10],#,100,1000], #number of trees, change it to 1000 for better results
              'silent': [1] #1 to suppress output
              #'missing':[-999],
              #'seed': [1337]
             }

# Loose grid search
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower, let this be 1 for now
              'objective': ['multi:softmax'], #indicates multiclass classification
              'learning_rate': np.linspace(0.1,1,10), #learning rate, aka 'eta' value, check 0.1 to 1
              'max_depth': np.linspace(3,10,8).astype(int), #maximum depth of a tree, check 3 to 10
              'min_child_weight': np.linspace(8,12,5), #tree parameters
              'gamma': [0,10],
              'subsample': np.linspace(0.5,1,6),
              'colsample_bytree': np.linspace(0.5,1,6),
              'reg_lambda': [1,10],
              'reg_alpha': [0],
              'n_estimators': [10],#,100,1000], #number of trees, change it to 1000 for better results
              'silent': [1] #1 to suppress output
              #'missing':[-999],
              #'seed': [1337]
             }

"""

# Tuning parameters for optimal values
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# 1. Fix learning rate and number of estimators
# 2. Tune max_depth and min_child_weight
# 3. Tune gamma
# 4. Tune subsample and colsample_bytree
# 5. Tune regularization parameters
# 6. Reduce learning and add more trees
# Difficult to get a very big leap in performance by just using parameter tuning or slightly better models
# Significant jump can be obtained by other methods like:
#   feature engineering, creating ensemble of models, stacking, etc

# Grid search with one value for each param is essentially same as no grid search
# Grid search actually may yield worse results because higher model score doesn't necessarily mean higher classification accuracy

# The "best" parameters that yield the highest classification accuracy
parameters = {#'nthread':[4], #when use hyperthread, xgboost may become slower, let this be 1 for now
              'objective': ['multi:softmax'], #indicates multiclass classification
              'learning_rate': [0.5], #learning rate, aka 'eta' value in 
              'n_estimators': [230], #number of trees, change it to 1000 for better results
              'max_depth': [10], #maximum depth of a tree
              'min_child_weight': [11],
              'gamma': [0.2],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'reg_lambda': [1],
              'silent': [1], #1 to suppress output
              #'missing':[-999],
              #'seed': [1337]
             }

# The "best" parameters returned from grid search
parameters = {#'nthread':[4], #when use hyperthread, xgboost may become slower, let this be 1 for now
              'objective': ['multi:softmax'], #indicates multiclass classification
              'learning_rate': [0.07], #learning rate, aka 'eta' value in 
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'max_depth': [6], #maximum depth of a tree
              'min_child_weight': [8],
              'gamma': [0],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'reg_lambda': [100],
              'silent': [1], #1 to suppress output
              #'missing':[-999],
              #'seed': [1337]
             }

# NOTE 1: Old scores and new scores are the same because same training data
# NOTE 2: Higher cross val score seems to yield worse results

# Trial 1 (orig params, learn rate 0.5 rounds 1000): 0.66249, Score 0.584 (Old: Accuracy 0.675, Score 0.584)
# rounds: 1000, learning_rate: 0.5, max_depth:10, min_child_weight:11, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:1 
# Conclusion: Accuracy high for a low score

# Trial 2 (new params after grid search, learn rate 0.07 rounds 1000): 0.63749, Score 0.61 (Old: Accuracy 0.655, Score 0.61)
# rounds: 1000, learning_rate: 0.07, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:100
# Conclusion: High score due to tuning parameters, but lower accuracy wtf

# Trial 3 (BEST) (gamma is god): 0.7, Score 0.579 (Old: Accuracy 0.7, Score 0.579)
# rounds: 230, learning_rate: 0.5, max_depth:10, min_child_weight:11, gamma:0.2, subsample:0.8, colsample_bytree:0.7, reg_lambda:1 
# Conclusion: Even lower cross val score but high prediction

# Experiment 0.1 (new params, learn rate 0.2 rounds 200): ?, Score 0.61 (Old: Accuracy 0.65, Score 0.61)
# rounds: 200, learning_rate: 0.2, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:100
# Conclusion: Highest score of 0.61, but accuracy decreases for old, new?

# Experiment 0.2 (new params, learn rate 0.07 rounds 1000): ?, Score 0.61 (Old: Accuracy 0.655, Score 0.61)
# rounds: 200, learning_rate: 0.2, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:100
# Conclusion: Highest score of 0.61, but accuracy decreases for old, new?

# Experiment 1 trial (new params, same learn rate and rounds): ?, Score 0.587 (Old: Accuracy 0.655, Score 0.587)
# rounds: 1000, learning_rate: 0.5, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:100
# Conclusion: Higher score, but accuracy decreases for old, new?

# Experiment 1.1 (dec learn rate to 0.01 from Experiment 1): ?, Score 0.593 (Old: Accuracy 0.625, Score 0.593, Training: 0.819)
# rounds: 1000, learning_rate: 0.01, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:100
# Conclusion: Decreases score, accuracy likely to decrease, bad

# Experiment 2 (old params, learn rate 0.2 rounds 200): ?, Score 0.593 (Old: Accuracy 0.645, Score 0.583)
# rounds: 200, learning_rate: 0.2, max_depth:10, min_child_weight:11, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:1 
# Conclusion: Doesn't give high score of 0.61 like for new params, bad

# Experiment 3 (new params, learn rate 0.5 rounds 1000, lambda 1000): ?, Score 0.607 WOW (Old: Accuracy 0.65, Score 0.607)
# rounds: 1000, learning_rate: 0.5, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:1000
# Conclusion: Higher score, but accuracy decreases for old, new?

# Train model
clf_XGB_GS = GridSearchCV(xgb_model, #classifier is XGBoost
                          parameters, #model parameters to perform grid search over
                          n_jobs=1,#number of jobs to run in parallel, leave at 1 now b/c shit breaks when rerunning
                          cv=5, #number of folds for cross validation
                          verbose=2, #display some output
                          refit=True #refit based on best params
                         )

# Fit model to data
clf_XGB_GS.fit(X_train, Y_train)
#print 'Grid scores: ', clf_XGB_GS.grid_scores_
#print 'Best params: ', clf_XGB_GS.best_params_
#print 'Best score: ', clf_XGB_GS.best_score_

# Make prediction
ypred_XGB_GS = clf_XGB_GS.predict(X_test) #on test set
ypred_on_train_XGB_GS = clf_XGB_GS.predict(X_train) #on training set


# In[6]:

### ~~~~~ XGBoost with Grid Search (Accuracy Statistics) ~~~~~ ###

print 'XGBoost with Grid Search:\n'

print 'Best score: ', clf_XGB_GS.best_score_
print 'Best params: ', clf_XGB_GS.best_params_
print 'Grid scores: ', clf_XGB_GS.grid_scores_
print
# 200 rounds: learning_rate: 0.2, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:100
# 0.585 max_depth:5, min_child_weight:11 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.5, 'min_child_weight': 11, 'n_estimators': 200, 'subsample': 0.8, 'objective': 'multi:softmax', 'max_depth': 5}
# 0.588 max_depth:6, min_child_weight:10 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.5, 'min_child_weight': 10, 'n_estimators': 200, 'subsample': 0.8, 'objective': 'multi:softmax', 'max_depth': 6} 
# 0.591 max_depth:6, min_child_weight:8 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.5, 'min_child_weight': 8, 'n_estimators': 200, 'subsample': 0.8, 'objective': 'multi:softmax', 'max_depth': 6}
# 0.591 gamma:0 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.5, 'min_child_weight': 8, 'n_estimators': 200, 'subsample': 0.8, 'objective': 'multi:softmax', 'max_depth': 6, 'gamma': 0}
# 0.591 subsample:0.8, colsample_bytree:0.7 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.5, 'min_child_weight': 8, 'n_estimators': 200, 'subsample': 0.8, 'objective': 'multi:softmax', 'max_depth': 6, 'gamma': 0}
# 0.608 reg_lambda:100 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.5, 'min_child_weight': 8, 'n_estimators': 200, 'subsample': 0.8, 'reg_lambda': 100, 'objective': 'multi:softmax', 'max_depth': 6, 'gamma': 0}
# 0.61 learning_rate: 0.2 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.2, 'min_child_weight': 8, 'n_estimators': 200, 'subsample': 0.8, 'reg_lambda': 100, 'objective': 'multi:softmax', 'max_depth': 6, 'gamma': 0}
# 1000 rounds:
# orig: 0.584 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.5, 'min_child_weight': 11, 'n_estimators': 1000, 'subsample': 0.8, 'objective': 'multi:softmax', 'max_depth': 10}
# new: 0.597 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.2, 'min_child_weight': 8, 'n_estimators': 1000, 'subsample': 0.8, 'reg_lambda': 100, 'objective': 'multi:softmax', 'max_depth': 6, 'gamma': 0}
# 0.602 learning_rate:0.1 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.2, 'min_child_weight': 8, 'n_estimators': 1000, 'subsample': 0.8, 'reg_lambda': 100, 'objective': 'multi:softmax', 'max_depth': 6, 'gamma': 0}
# 0.61 learning_rate:0.07 {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.07, 'min_child_weight': 8, 'n_estimators': 1000, 'subsample': 0.8, 'reg_lambda': 100, 'objective': 'multi:softmax', 'max_depth': 6, 'gamma': 0}
# 0.592 learning_rate:0.3 lambda:1000
# 0.603 learning_rate:0.5 lambda:1000

train_error_XGB_GS = np.sum(np.equal(ypred_on_train_XGB_GS,Y_train))/float(ypred_on_train_XGB_GS.size)
print 'Training percent correct: ', train_error_XGB_GS

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_XGB_GS = np.sum(np.equal(ypred_XGB_GS+1,Y_sample))/float(ypred_XGB_GS.size)
print 'Testing percent correct: ', test_error_XGB_GS

print Y_sample.astype(int)
print ypred_XGB_GS.astype(int)+1


# In[7]:

### ~~~~~ XGBoost with Grid Search (Cross-Validation) ~~~~~ ###

# Use best parameters found in grid search
best_params = clf_XGB_GS.best_params_
clf_XGB_GS_CV = xgb.XGBClassifier(objective=best_params['objective'],
                                  learning_rate=best_params['learning_rate'],
                                  n_estimators=best_params['n_estimators'],
                                  max_depth=best_params['max_depth'],
                                  min_child_weight=best_params['min_child_weight'],
                                  gamma=best_params['gamma'],
                                  subsample=best_params['subsample'],
                                  colsample_bytree=best_params['colsample_bytree'],
                                  reg_lambda=best_params['reg_lambda'],
                                  silent=best_params['silent']
                                 )

# Get cross-validation score
scores_XGB_GS_CV = cross_val_score(clf_XGB_GS_CV, X_train, Y_train, cv=5, verbose=0)
print 'Cross-validation score: ', np.mean(scores_XGB_GS_CV)
print 'Scores for all folds: ', scores_XGB_GS_CV


# In[8]:

### ~~~~~ Logistic Regression with and without L1 Regularization (Testing and Training) ~~~~~ ###

# Turn grid search on or off, if you've already found the optimum values just turn it off for speed
grid_search_off_log_reg = True
grid_search_off_log_reg_11 = True

# Without regularization

# Train and fit model to data
if grid_search_off_log_reg:
    clf_log_reg = LogisticRegression(penalty='l1',C=10000,fit_intercept=True).fit(X_train,Y_train)
        # Make C high to make regularization strength weak    
else:
    log_reg_model = LogisticRegression(penalty='l1',fit_intercept=True)
    # Grid search just to find score
    parameters = {'C':[10000]}
    clf_log_reg = GridSearchCV(log_reg_model,
                               parameters,
                               cv=5, # 5-fold cross-validation
                               verbose=2, #display some output
                               refit=True #refit based on best params
                              )
    clf_log_reg.fit(X_train,Y_train)
    
# Predict
ypred_log_reg = clf_log_reg.predict(X_test) #on test set
ypred_on_train_log_reg = clf_log_reg.predict(X_train) #on training set

# C=100000 - 0.916, score 0.505
# C=10000 - 0.914, score 0.507
# C=1000 - 0.916, score 0.505
# C=100 - 0.915, score 0.505
# C=10 - 0.915, score 0.503
# C=1 - 0.914, score 0.508


# With regularization

# Train and fit model to data
if grid_search_off_log_reg_11:
    clf_log_reg_l1 = LogisticRegression(penalty='l1',C=0.01,fit_intercept=True).fit(X_train,Y_train)
else:
    log_reg_l1_model = LogisticRegression(penalty='l1',fit_intercept=True)
    parameters = {'C':[0.01]}
    clf_log_reg_l1 = GridSearchCV(log_reg_l1_model,
                                  parameters,
                                  cv=5, # 5-fold cross-validation
                                  verbose=2, #display some output
                                  refit=True #refit based on best params
                                 )
    clf_log_reg_l1.fit(X_train,Y_train)
    
# Predict
ypred_log_reg_L1 = clf_log_reg_l1.predict(X_test) #on test set
ypred_on_train_log_reg_L1 = clf_log_reg_l1.predict(X_train) #on training set

# C=0.1 - 0.888, score 0.539
# C=0.01 - 0.811, score 0.594
# C=0.001 - 0.667, score 0.589


# In[9]:

### ~~~~~ Logistic Regression with and without L1 Regularization (Accuracy Statistics) ~~~~~ ###

print 'Logistic Regression:\n'

# No reg
if not grid_search_off_log_reg:
    print 'Best score: ', clf_log_reg.best_score_
    print 'Best params: ', clf_log_reg.best_params_
    print 'Grid scores: ', clf_log_reg.grid_scores_
    print
    #Grid scores:  [mean: 0.50500, std: 0.04604, params: {'penalty': 'l1', 'C': 1000}, mean: 0.50700, std: 0.04445, params: {'penalty': 'l1', 'C': 10000}, mean: 0.50500, std: 0.04604, params: {'penalty': 'l1', 'C': 100000}]
    #Best params:  {'penalty': 'l1', 'C': 10000}
    #Best score:  0.507

train_error_log_reg = np.sum(np.equal(ypred_on_train_log_reg,Y_train))/float(ypred_on_train_log_reg.size)
print 'Training percent correct: ', train_error_log_reg

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_log_reg = np.sum(np.equal(ypred_log_reg+1,Y_sample))/float(ypred_log_reg.size)
print 'Testing percent correct: ', test_error_log_reg

print Y_sample.astype(int)
print ypred_log_reg.astype(int)+1
print

# L1 reg

if not grid_search_off_log_reg_11:
    print 'Best score: ', clf_log_reg_l1.best_score_
    print 'Best params: ', clf_log_reg_l1.best_params_
    print 'Grid scores: ', clf_log_reg_l1.grid_scores_
    print
    #Grid scores:  [mean: 0.50500, std: 0.04483, params: {'penalty': 'l1', 'C': 100}, mean: 0.50300, std: 0.04578, params: {'penalty': 'l1', 'C': 10}, mean: 0.50800, std: 0.03855, params: {'penalty': 'l1', 'C': 1}, mean: 0.53900, std: 0.05517, params: {'penalty': 'l1', 'C': 0.1}, mean: 0.59400, std: 0.04705, params: {'penalty': 'l1', 'C': 0.01}, mean: 0.58900, std: 0.04341, params: {'penalty': 'l1', 'C': 0.001}]
    #Best params:  {'penalty': 'l1', 'C': 0.01}
    #Best score:  0.594

train_error_log_reg_L1 = np.sum(np.equal(ypred_on_train_log_reg_L1,Y_train))/float(ypred_on_train_log_reg_L1.size)
print 'Training percent correct (L1): ', train_error_log_reg_L1

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_log_reg_L1 = np.sum(np.equal(ypred_log_reg_L1+1,Y_sample))/float(ypred_log_reg_L1.size)
print 'Testing percent correct (L1): ', test_error_log_reg_L1

print Y_sample.astype(int)
print ypred_log_reg_L1.astype(int)+1


# In[10]:

### ~~~~~ Logistic Regression with and without L1 Regularization (Cross-Validation) ~~~~~ ###

# Without regularization

# Train and fit
clf_log_reg_CV = LogisticRegression(penalty='l1',C=10000,fit_intercept=True)
    # Make C high to make regularization strength weak

# Get cross-validation score
scores_log_reg_CV = cross_val_score(clf_log_reg_CV, X_train, Y_train, cv=5, verbose=0)
print 'Cross-validation score (no regularization): ', np.mean(scores_log_reg_CV)
print 'Scores for all folds (no regularization): ', scores_log_reg_CV
print

# With regularization

# Train and fit
clf_log_reg_l1_CV = LogisticRegression(penalty='l1',C=0.01,fit_intercept=True)

# Get cross-validation score
scores_log_reg_l1_CV = cross_val_score(clf_log_reg_l1_CV, X_train, Y_train, cv=5, verbose=0)
print 'Cross-validation score (L1 regularization): ', np.mean(scores_log_reg_l1_CV)
print 'Scores for all folds (L1 regularization): ', scores_log_reg_l1_CV


# In[11]:

### ~~~~~ SVMs with Grid Search (Training and Testing) ~~~~~ ###

# Classifier to perform grid search for is SVM
svm_model = SVC()

# Parameters to perform grid search over
# Use values outlined in paper for loose grid search:
# C: 2^-5, 2^-3, ... 2^15
# gamma: 2^-15, ... 2^3
C_exponents = np.linspace(-5,15,11)
gamma_exponents = np.linspace(-15,3,10)

"""
# Grid search with all kernels
# Hangs because of poly with degree > 1
parameters = {'kernel': ['linear','poly','rbf','sigmoid'],
              'gamma': np.power(2,gamma_exponents),
              'degree': [1]#np.linspace(1,5,1),
              'C': np.power(2,C_exponents)
             }

"""

# Grid search with one kernel
# Poly hangs for degrees greater than 1, why?
parameters = {'kernel': ['poly'],
              'gamma': np.power(2,gamma_exponents),
              'degree': [1],
              'C': np.power(2,C_exponents)
             }

# Grid search
clf_SVM_GS = GridSearchCV(svm_model,
                          parameters,
                          cv=5, # 5-fold cross-validation
                          verbose=2, #display some output
                          refit=True #refit based on best params
                         )

# Fit model to training data
clf_SVM_GS.fit(X_train, Y_train)

# Make prediction
ypred_SVM_GS = clf_SVM_GS.predict(X_test) #on test set
ypred_on_train_SVM_GS = clf_SVM_GS.predict(X_train) #on training set


# In[12]:

### ~~~~~ SVMs with Grid Search (Accuracy Statistics) ~~~~~ ###

print 'SVMs with Grid Search:\n'

print 'Best score: ', clf_SVM_GS.best_score_
print 'Best params: ', clf_SVM_GS.best_params_
print 'Grid scores: ', clf_SVM_GS.grid_scores_
print
# rbf best score/params:  0.349 {'kernel': 'rbf', 'C': 0.03125, 'gamma': 0.00048828125}
# linear kernel best scores/params:  0.54 {'kernel': 'linear', 'C': 0.03125, 'gamma': 3.0517578125e-05}
# sigmoid kernel best scores/params: 0.1 {'kernel': 'sigmoid', 'C': 0.03125, 'gamma': 3.0517578125e-05}
# poly kernel best scores/params: 0.621 {'kernel': 'poly', 'C': 0.03125, 'gamma': 3.0517578125e-05, 'degree': 1}

train_error_SVM_GS = np.sum(np.equal(ypred_on_train_SVM_GS,Y_train))/float(ypred_on_train_SVM_GS.size)
print 'Training percent correct: ', train_error_SVM_GS

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_SVM_GS = np.sum(np.equal(ypred_SVM_GS+1,Y_sample))/float(ypred_SVM_GS.size)
print 'Testing percent correct: ', test_error_SVM_GS

print Y_sample.astype(int)
print ypred_SVM_GS.astype(int)+1


# In[13]:

### ~~~~~ SVMs with Grid Search (Cross-Validation) ~~~~~ ###

# Use best parameters found in grid search
best_params = clf_SVM_GS.best_params_
clf_SVM_GS_CV = SVC(kernel=best_params['kernel'],
                    C=best_params['C'],
                    gamma=best_params['gamma'],
                    degree=best_params['degree']
                   )

# Get cross-validation score
# Get cross-validation score
scores_SVM_GS_CV = cross_val_score(clf_SVM_GS_CV, X_train, Y_train, cv=5, verbose=0)
print 'Cross-validation score: ', np.mean(scores_SVM_GS_CV)
print 'Scores for all folds: ', scores_SVM_GS_CV


# In[14]:

### ~~~~~ Gaussian Generative Model using Naive Bayes (Training and Testing) ~~~~~ ###

# Train model
gnb = GaussianNB()

# Fit model to data and predict
ypred_GNB = gnb.fit(X_train, Y_train).predict(X_test) # predict on test data
ypred_on_train_GNB = gnb.fit(X_train, Y_train).predict(X_train) # predict on training data

# Get cross-validation score
scores_GNB_CV = cross_val_score(gnb, X_train, Y_train, cv=5)

# Get score using grid search (should match with cross-val score)
parameters = {}
clf_GNB = GridSearchCV(gnb,
                       parameters,
                       cv=5, # 5-fold cross-validation
                       verbose=0, #display some output
                       refit=True #refit based on best params
                      )
clf_GNB.fit(X_train, Y_train)


# In[15]:

### ~~~~~ Gaussian Generative Model using Naive Bayes (Accuracy Statistics + Cross-Validation) ~~~~~ ###

print 'Gaussian Generative Model using Naive Bayes:\n'

print 'Cross-validation score', np.mean(scores_GNB_CV) #averaging the 5 folds
print 'Grid-search score: ', clf_GNB.best_score_
print 'Scores for all folds: ', scores_GNB_CV
# Note: cross-validation score and grid-search score for one param set is the SAME
print

train_error_GNB = np.sum(np.equal(ypred_on_train_GNB,Y_train))/float(ypred_on_train_GNB.size)
print 'Training percent correct: ', train_error_GNB

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_GNB = np.sum(np.equal(ypred_GNB+1,Y_sample))/float(ypred_GNB.size)
print 'Testing percent correct: ', test_error_GNB

print Y_sample.astype(int)
print ypred_GNB.astype(int)+1


# In[16]:

### ~~~~~ Gaussian Generative Model, Manual Implementation (Training and Testing) ~~~~~ ###

# 4.75: Mean for class k
def classMean(X,Y,k):
    class_indices = np.where(np.equal(Y,k))
    class_indices = class_indices[0]
    return np.mean(X[class_indices,:],axis=0)

# 4.79-4.80: Covariance for class k
def classCovariance(X,Y,k):
    class_indices = np.where(np.equal(Y,k))
    class_indices = class_indices[0]
    return np.cov(X[class_indices,:],rowvar=False)
        # if rowvar is True, then each row is a variable, columns are observations
        # Since our columns are features, we make rowvar False
        
# 4.78: Weighted average for class covariances
def weightedCovariance(X,Y,K):
    # Number of observations and features
    N = X.shape[0]
    M = X.shape[1]
    # Get row indices in X where corresponding Y's are of a certain class
    class_indices = [0 for k in range(K)]
    for k in range(K):
        class_indices[k] = np.where(np.equal(Y,k)) #labels in data matrix start with 0 as well, so don't add 1 to k
        class_indices[k] = class_indices[k][0]
    # Get number of observations for each class
    Nk = [0 for k in range(K)]
    #N_check = 0
    for k in range(K):
        Nk[k] = Y[class_indices[k]].shape[0]
        #N_check += Nk[k]
    # Find covariances for each class and build weighted covariance
    cov = np.zeros((M,M))
    for k in range(K):
        cov += float(Nk[k])/N * classCovariance(X,Y,k)   
    return cov

# Train model by finding class means and covariance from training data X,Y
def trainGenerativeModel(X,Y,K):

    # Number of classes, number of observations
    N = X_train.shape[0]

    # Get means and for each class using maximum likelihood approach
    mu = [0 for k in range(K)]
    for k in range(K):
        mu[k] = classMean(X_train,Y_train,k) # Note: we subtracted 1 from Y_train, so classes start at 0
        
    # Get weighted class covariance using maximum likelihood approach
    cov = weightedCovariance(X_train,Y_train,K)
    
    # Freeze the pdf for each class so you don't have to regenerate it later when testing
    frozenpdf = [0 for k in range(K)]
    for k in range(K):
        frozenpdf[k] = multivariate_normal(mu[k],cov)
        
    return mu, cov, frozenpdf

# Fit model to data set X
def fitGenerativeModel(X,frozenpdf,K):

    # Number of observations in dataset
    N = X.shape[0]

    # Get probabilities of each class for each observation
    normpdf = [np.zeros(N) for k in range(K)]
    for k in range(K):
        for n in range(N):
            normpdf[k][n] = frozenpdf[k].pdf(X[n,:])

    # For each xn, find class associated with max probability
    max_class = [-1 for n in range(N)]
    for n in range(N):
        max_prob = -1
        for k in range(K):
            if normpdf[k][n] > max_prob:
                max_prob = normpdf[k][n]
                max_class[n] = k

    # Convert list to np array
    ypred = np.array(max_class)
    
    return ypred

# Train and fit model
K = 10
(mu, cov, frozenpdf) = trainGenerativeModel(X_train,Y_train,K)
ypred_gen_model = fitGenerativeModel(X_test,frozenpdf,K) #on test data
ypred_on_train_gen_model = fitGenerativeModel(X_train,frozenpdf,K) #on training data


# In[17]:

### ~~~~~ Gaussian Generative Model, Manual Implementation (Accuracy Statistics) ~~~~~ ###

print 'Gaussian Generative Model, Manual Implementation:\n'

train_error_gen_model = np.sum(np.equal(ypred_on_train_gen_model,Y_train))/float(ypred_on_train_gen_model.size)
print 'Training percent correct: ', train_error_gen_model

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_gen_model = np.sum(np.equal(ypred_gen_model+1,Y_sample))/float(ypred_gen_model.size)
print 'Testing percent correct: ', test_error_gen_model

print Y_sample.astype(int)
print ypred_gen_model.astype(int)+1


# In[18]:

### ~~~~~ Gaussian Generative Model, Manual Implementation (Cross-Validation) ~~~~~ ###

folds = 5

# Splitting strategy
#kf = KFold(n_splits=folds) # won't work because the class labels are ordered in training set
kf = StratifiedKFold(n_splits=folds)

i = 0
scores_gen_model_CV = []
for train_index, test_index in kf.split(X_train,Y_train):
        
    # Split data
    X_train_split, X_test_split = X_train[train_index], X_train[test_index]
    Y_train_split, Y_test_split = Y_train[train_index], Y_train[test_index]

    # Train model
    K = 10
    (mu, cov, frozenpdf) = trainGenerativeModel(X_train_split,Y_train_split,K)

    # Predict using model
    ypred_gen_model_CV = fitGenerativeModel(X_test_split,frozenpdf,K) #on test data
    ypred_on_train_gen_model_CV = fitGenerativeModel(X_train_split,frozenpdf,K) #on training data

    # Print error for splt predicting testing and training sets
    # Error is equivalent to score in GridSearch? Makes sense then, since GridSearch uses CV...
    print ('Fold %d:' % (i+1))
    train_error_gen_model_CV = np.sum(np.equal(ypred_on_train_gen_model_CV,Y_train_split))/float(ypred_on_train_gen_model_CV.size)
    print 'Training percent correct: ', train_error_gen_model_CV
    test_error_gen_model_CV = np.sum(np.equal(ypred_gen_model_CV,Y_test_split))/float(ypred_gen_model_CV.size)
    scores_gen_model_CV.append(test_error_gen_model_CV)
    print 'Testing percent correct: ', test_error_gen_model_CV
    i += 1

print '\nCross-validation score (mean of all folds): ', np.mean(scores_gen_model_CV)


# In[19]:

### ~~~~~ Save all results ~~~~~ ###

ypred_list = [ypred_XGB, ypred_XGB_GS, ypred_log_reg, ypred_log_reg_L1, ypred_SVM_GS, ypred_GNB, ypred_gen_model]
model_names = ['XGB', 'XGB_GS', 'log_reg', 'log_reg_L1', 'SVM_GS', 'GNB', 'gen_model']

# Go through each model prediction and output into file
for i in range(len(ypred_list)):
    
    # Current using results from XGBoost + Grid Search (even though the Grid Search is technically useless)
    ypred = ypred_list[i]
    print model_names[i], ypred.astype(int)+1

    #Save results to CSV
    N = X_test.shape[0]
    observation_number = np.arange(N)+1 # first column are numbered obsevations, predictions are second column
    output = np.transpose(np.stack((observation_number, ypred+1))).astype(int) # remember to add back 1 to labels
    np.savetxt(data_folder + "sampleSubmission" + model_names[i] + ".csv", output, fmt='%i', delimiter=",", header='Id,Prediction', comments='')
        # comments='' gets rid of hash mark


# In[ ]:

### ~~~~~ Save results for one model ~~~~~ ###

ypred = ypred_XGB
print ypred.astype(int)+1

#Save results to CSV
N = X_test.shape[0]
observation_number = np.arange(N)+1 # first column are numbered obsevations, predictions are second column
output = np.transpose(np.stack((observation_number, ypred+1))).astype(int) # remember to add back 1 to labels
np.savetxt(data_folder + "sampleSubmission.csv", output, fmt='%i', delimiter=",", header='Id,Prediction', comments='')
    # comments='' gets rid of hash mark
    


# In[ ]:



