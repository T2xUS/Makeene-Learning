
# coding: utf-8

# In[1]:

import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV # use Grid Search to select best XGB model using CV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.cross_validation import train_test_split # for cross validation
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
#print X_train, X_train.shape, type(X_train)
#print Y_train, Y_train.shape, type(Y_train)
#print X_test, X_test.shape, type(X_test)


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

### ~~~~~ XGBoost on training set split into training and testing (cross validation) ~~~~~ ###

# Why do I get the same accuracies for multiple iterations despite this being a random split?
X_train_split,X_test_split,Y_train_split,Y_test_split = train_test_split(X_train, Y_train, 
                                                                         test_size=0.2, #inverse of number of folds
                                                                         random_state=42,
                                                                        )
# Load data from np matrix into XGBoost
# Can alternative load data from files directly
dtrain = xgb.DMatrix(X_train_split, label=Y_train_split)

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
dtest = xgb.DMatrix(X_test_split, label=Y_test_split)
ypred_XGB = bst.predict(dtest) #on test set
ypred_on_train_XGB = bst.predict(dtrain) #on training set

# Print error for splt predicting testing and training sets
# Error is equivalent to score in GridSearch? Makes sense then, since GridSearch uses CV...
train_error_XGB = np.sum(np.equal(ypred_on_train_XGB,Y_train_split))/float(ypred_on_train_XGB.size)
print 'Training percent correct: ', train_error_XGB
test_error_XGB = np.sum(np.equal(ypred_XGB,Y_test_split))/float(ypred_XGB.size)
print 'Testing percent correct: ', test_error_XGB


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

# NOTE 1: Old scores and new scores are the same because same training data
# NOTE 2: Higher cross val score seems to yield worse results

# Trial 1 (BEST) (orig params, learn rate 0.5 rounds 1000): 0.66249, Score 0.584 (Old: Accuracy 0.675, Score 0.584)
# rounds: 1000, learning_rate: 0.5, max_depth:10, min_child_weight:11, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:1 
# Conclusion: Accuracy high for a low score

# Trial 2 (new params after grid search, learn rate 0.07 rounds 1000): 0.63749, Score 0.61 (Old: Accuracy 0.655, Score 0.61)
# rounds: 200, learning_rate: 0.2, max_depth:6, min_child_weight:8, gamma:0, subsample:0.8, colsample_bytree:0.7, reg_lambda:100
# Conclusion: High score due to tuning parameters, but lower accuracy wtf

# Trial 3 (gamma is god): ?, Score 0.579 (Old: Accuracy 0.7, Score 0.579)
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
clf = GridSearchCV(xgb_model, #classifier is XGBoost
                   parameters, #model parameters to perform grid search over
                   n_jobs=1,#number of jobs to run in parallel, leave at 1 now b/c shit breaks when rerunning
                   cv=5, #number of folds for cross validation
                   verbose=2, #display some output
                   refit=True #refit based on best params
                  )

# Fit model to data
clf.fit(X_train, Y_train)
#print 'Grid scores: ', clf.grid_scores_
print 'Best params: ', clf.best_params_
print 'Best score: ', clf.best_score_

# Make prediction
ypred_XGB_GS = clf.predict(X_test) #on test set
ypred_on_train_XGB_GS = clf.predict(X_train) #on training set


# In[6]:

### ~~~~~ XGBoost with Grid Search (Accuracy Statistics) ~~~~~ ###

print 'XGBoost with Grid Search:\n'

print 'Grid scores: ', clf.grid_scores_
print 'Best params: ', clf.best_params_
print 'Best score: ', clf.best_score_
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

### ~~~~~ Logistic Regression with and without L1 Regularization (Testing and Training) ~~~~~ ###

# Without regularization
clf = LogisticRegression(penalty='l1',C=1000,fit_intercept=True).fit(X_train,Y_train)
    # Make C high to make regularization strength weak
ypred_log_reg = clf.predict(X_test) #on test set
ypred_on_train_log_reg = clf.predict(X_train) #on training set

# With regularization
# TO DO: Tweak regularization params C
clf = LogisticRegression(penalty='l1',C=0.1,fit_intercept=True).fit(X_train,Y_train)
ypred_log_reg_L1 = clf.predict(X_test) #on test set
ypred_on_train_log_reg_L1 = clf.predict(X_train) #on training set


# In[8]:

### ~~~~~ Logistic Regression with and without L1 Regularization (Accuracy Statistics) ~~~~~ ###

print 'Logistic Regression:\n'

train_error_log_reg = np.sum(np.equal(ypred_on_train_log_reg,Y_train))/float(ypred_on_train_log_reg.size)
print 'Training percent correct: ', train_error_log_reg

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_log_reg = np.sum(np.equal(ypred_log_reg+1,Y_sample))/float(ypred_log_reg.size)
print 'Testing percent correct: ', test_error_log_reg

print Y_sample.astype(int)
print ypred_log_reg.astype(int)+1

train_error_log_reg_L1 = np.sum(np.equal(ypred_on_train_log_reg_L1,Y_train))/float(ypred_on_train_log_reg_L1.size)
print 'Training percent correct (L1): ', train_error_log_reg_L1

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_log_reg_L1 = np.sum(np.equal(ypred_log_reg_L1+1,Y_sample))/float(ypred_log_reg_L1.size)
print 'Testing percent correct (L1): ', test_error_log_reg_L1

print Y_sample.astype(int)
print ypred_log_reg_L1.astype(int)+1


# In[9]:

### ~~~~~ SVMs with Grid Search (Training and Testing) ~~~~~ ###

# Classifier to perform grid search for is SVM
svm_model = SVC()

# Parameters to perform grid search over
# Use values outlined in paper for loose grid search:
# C: 2^-5, 2^-3, ... 2^15
# gamma: 2^-15, ... 2^3

C_exponents = np.linspace(-5,15,11)
gamma_exponents = np.linspace(-15,3,10)

# Grid search with all kernels
parameters = {'kernel': ['linear','poly','rbf','sigmoid'],
              'gamma': np.power(2,gamma_exponents),
              'C': np.power(2,C_exponents)
             }

# Grid search with one kernel
parameters = {'kernel': ['rbf'],
              'gamma': np.power(2,gamma_exponents),
              'C': np.power(2,C_exponents)
             }

# From last year's Kaggle
parameters = {'kernel': ['rbf'],
              'gamma': np.arange(.001,.1,.005),
              'C': range(1,1000,100)
              }

# Train model
clf = GridSearchCV(svm_model,
                   parameters,
                   cv=5, # 5-fold cross-validation
                   verbose=2, #display some output
                   refit=True #refit based on best params
                  )

# Fit model to data
clf.fit(X_train, Y_train)

# Make prediction
ypred_SVM_GS = clf.predict(X_test) #on test set
ypred_on_train_SVM_GS = clf.predict(X_train) #on training set


# In[10]:

### ~~~~~ SVMs with Grid Search (Accuracy Statistics) ~~~~~ ###

print 'SVMs with Grid Search:\n'

train_error_SVM_GS = np.sum(np.equal(ypred_on_train_SVM_GS,Y_train))/float(ypred_on_train_SVM_GS.size)
print 'Training percent correct: ', train_error_SVM_GS

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_SVM_GS = np.sum(np.equal(ypred_SVM_GS+1,Y_sample))/float(ypred_SVM_GS.size)
print 'Testing percent correct: ', test_error_SVM_GS

print Y_sample.astype(int)
print ypred_SVM_GS.astype(int)+1


# In[11]:

### ~~~~~ Gaussian Generative Model using Naive Bayes (Training and Testing) ~~~~~ ###

gnb = GaussianNB()
ypred_GNB = gnb.fit(X_train, Y_train).predict(X_test) # predict on test data
ypred_on_train_GNB = gnb.fit(X_train, Y_train).predict(X_train) # predict on training data


# In[12]:

### ~~~~~ Gaussian Generative Model using Naive Bayes (Accuracy Statistics) ~~~~~ ###

print 'Gaussian Generative Model using Naive Bayes:\n'

train_error_GNB = np.sum(np.equal(ypred_on_train_GNB,Y_train))/float(ypred_on_train_GNB.size)
print 'Training percent correct: ', train_error_GNB

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_GNB = np.sum(np.equal(ypred_GNB+1,Y_sample))/float(ypred_GNB.size)
print 'Testing percent correct: ', test_error_GNB

print Y_sample.astype(int)
print ypred_GNB.astype(int)+1


# In[13]:

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
ypred_on_train_gen_model = fitGenerativeModel(X_train,frozenpdf,10) #on training data


# In[14]:

### ~~~~~ Gaussian Generative Model, Manual Implementation (Accuracy Statistics) ~~~~~ ###

print 'Gaussian Generative Model, Manual Implementation:\n'

train_error_gen_model = np.sum(np.equal(ypred_on_train_gen_model,Y_train))/float(ypred_on_train_gen_model.size)
print 'Training percent correct: ', train_error_gen_model

sample_solution = np.loadtxt(data_folder + 'sample_solution.csv',dtype=int,delimiter=',',skiprows=1)
Y_sample = sample_solution[:,1]
test_error_gen_model = np.sum(np.equal(ypred_gen_model+1,Y_sample))/float(ypred_gen_model.size)
print 'Testing percent correct: ', test_error_gen_model

print Y_sample.astype(int)
print ypred_GNB.astype(int)+1


# In[15]:

### ~~~~~ Select Best Model and Save Results ~~~~~ ###

# Current using results from XGBoost + Grid Search (even though the Grid Search is technically useless)
ypred = ypred_XGB_GS
print ypred+1

#Save results to CSV
N = X_test.shape[0]
observation_number = np.arange(N)+1 # first column are numbered obsevations, predictions are second column
output = np.transpose(np.stack((observation_number, ypred+1))).astype(int) # remember to add back 1 to labels
np.savetxt(data_folder + "sampleSubmission.csv", output, fmt='%i', delimiter=",", header='Id,Prediction', comments='')
    # comments='' gets rid of hash mark


# In[ ]:



