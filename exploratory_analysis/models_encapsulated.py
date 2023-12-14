##############################################################################
# Based on code from the 'Hands-On-Genetic-Algorithms-with-Python' repository
# Author: Eyal Wirsansky, ai4java
# Link: https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter08
# Link: https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter09
##############################################################################

# 3rd party
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

# standard library
from math import floor

# internal imports
try:
    from util import make_difference_features, make_multiplicative_features
except:
    from exploratory_analysis.util import make_difference_features, make_multiplicative_features

# constants
RANDOM_SEED = 42
TEST_SIZE = 0.20
NUM_FOLDS = 3


########################
# Random Forest Search
########################
class HyperparameterTuningRandomForest:

    def __init__(self, randomSeed: int=RANDOM_SEED, testSize: float=TEST_SIZE, numFolds: int=NUM_FOLDS):
        
        self.randomSeed = randomSeed
        self.testSize = testSize
        self.initMachineFailureDataset()
        self.kfold = model_selection.StratifiedKFold(n_splits=numFolds)

    def initMachineFailureDataset(self):
        try:
            data = pd.read_csv('../machine failure.csv')
        except:
            data = pd.read_csv('machine failure.csv')

        # if the failure is not a twf, hdf, pwf, or osf then it is treated as a non failure
        data['Machine failure'] = np.where((data['TWF'] == 1) | (data['HDF'] == 1) | (data['PWF'] == 1) | (data['OSF'] == 1), 1, 0)
        data.drop(['UDI','Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)
        quality_map = {'L': 1, 'M': 2, 'H': 3}
        data['Type'] = data['Type'].map(quality_map)

        # add features
        subtraction_columns = make_difference_features(data.drop(columns=['Machine failure', 'Type']))
        multiplication_columns = make_multiplicative_features(data.drop(columns=['Machine failure', 'Type']))
        data = pd.concat([data, subtraction_columns, multiplication_columns], axis=1)

        # split and standardize the data
        X = data.drop(columns='Machine failure')
        y = data['Machine failure']
        self.X, _, self.y, _ = model_selection.train_test_split(X, y, test_size=self.testSize, random_state=self.randomSeed, stratify=y)

        scaler = StandardScaler()
        columns_to_scale = list(self.X.columns)
        columns_to_scale.remove('Type')

        self.X[columns_to_scale] = scaler.fit_transform(self.X[columns_to_scale])

    # RandomForestClassifier [min_samples_leaf, max_features, max_depth, n_estimators]:
    # min_samples_leaf: 1-25
    # max_features: 0.01-1.0
    # max_depth: 1-20
    # n_estimators: 50-200
    def convertParams(self, params):
        min_samples_leaf = round(params[0])
        max_features = round(params[1], 2)   # make value more coarse
        max_depth = round(params[2])
        n_estimators = round(params[3])
        return min_samples_leaf, max_features, max_depth, n_estimators

    def getAccuracy(self, params):
        samples, features, depth, estimators = self.convertParams(params)
        self.classifier = RandomForestClassifier(random_state=self.randomSeed,
                                                 n_jobs=-1,
                                                 min_samples_leaf=samples,
                                                 max_features=features,
                                                 max_depth=depth,
                                                 n_estimators=estimators
                                                )

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy',
                                                     n_jobs=-1)
        return cv_results.mean()

    def formatParams(self, params):
        samples, features, depth, estimators = self.convertParams(params)
        return f'min_samples_leaf={samples}, max_features={features}, max_depth={depth}, n_estimators={estimators}'


########################
# Decision Tree Search
########################
class HyperparameterTuningDecisionTree:

    def __init__(self, randomSeed: int=RANDOM_SEED, testSize: float=TEST_SIZE, numFolds: int=NUM_FOLDS):
        
        self.randomSeed = randomSeed
        self.testSize = testSize
        self.initMachineFailureDataset()
        self.kfold = model_selection.StratifiedKFold(n_splits=numFolds)

    def initMachineFailureDataset(self):
        try:
            data = pd.read_csv('../machine failure.csv')
        except:
            data = pd.read_csv('machine failure.csv')

        # if the failure is not a twf, hdf, pwf, or osf then it is treated as a non failure
        data['Machine failure'] = np.where((data['TWF'] == 1) | (data['HDF'] == 1) | (data['PWF'] == 1) | (data['OSF'] == 1), 1, 0)
        data.drop(['UDI','Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)
        quality_map = {'L': 1, 'M': 2, 'H': 3}
        data['Type'] = data['Type'].map(quality_map)

        # add features
        subtraction_columns = make_difference_features(data.drop(columns=['Machine failure', 'Type']))
        multiplication_columns = make_multiplicative_features(data.drop(columns=['Machine failure', 'Type']))
        data = pd.concat([data, subtraction_columns, multiplication_columns], axis=1)

        # split and standardize the data
        X = data.drop(columns='Machine failure')
        y = data['Machine failure']
        self.X, _, self.y, _ = model_selection.train_test_split(X, y, test_size=self.testSize, random_state=self.randomSeed, stratify=y)

        scaler = StandardScaler()
        columns_to_scale = list(self.X.columns)
        columns_to_scale.remove('Type')

        self.X[columns_to_scale] = scaler.fit_transform(self.X[columns_to_scale])

    # boundaries for DecisionTreeClassifier
    # max_depth: 1-25
    # min_samples_split: 0.01-1.0
    # min_samples_leaf: 1-30
    # max_features: 0.01-1.0
    # min_impurity_decrease: 0.001-0.03
    # [max_depth, min_samples_split, min_samples_leaf, max_features, min_impurity_decrease]
    def convertParams(self, params):
        max_depth = round(params[0])
        min_samples_split = round(params[1], 2)     # make value more coarse 
        min_samples_leaf = round(params[2])
        max_features = round(params[3], 2)          # make value more coarse
        min_impurity_decrease = round(params[4], 4) # make value more coarse
        return max_depth, min_samples_split, min_samples_leaf, max_features, min_impurity_decrease

    def getAccuracy(self, params):
        depth, samples_split, samples_leaf, features, impurity_decrease = self.convertParams(params)
        self.classifier = RandomForestClassifier(random_state=self.randomSeed,
                                                 n_jobs=-1,
                                                 max_depth=depth,
                                                 min_samples_split=samples_split,
                                                 min_samples_leaf=samples_leaf,
                                                 max_features=features,
                                                 min_impurity_decrease=impurity_decrease
                                                )

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy',
                                                     n_jobs=-1)
        return cv_results.mean()

    def formatParams(self, params):
        depth, samples_split, samples_leaf, features, impurity_decrease = self.convertParams(params)
        return f'max_depth={depth}, min_samples_split={samples_split}, min_samples_leaf={samples_leaf}, max_features={features}, min_impurity_decrease={impurity_decrease}'


########################
# MLP Search
########################
class HyperparameterTuningMlp:

    def __init__(self, randomSeed: int=RANDOM_SEED, testSize: float=TEST_SIZE, numFolds: int=NUM_FOLDS):

        self.randomSeed = randomSeed
        self.testSize = testSize
        self.initMachineFailureDataset()
        self.kfold = model_selection.StratifiedKFold(n_splits=numFolds)

    def initMachineFailureDataset(self):
        try:
            data = pd.read_csv('../machine failure.csv')
        except:
            data = pd.read_csv('machine failure.csv')

        # if the failure is not a twf, hdf, pwf, or osf then it is treated as a non failure
        data['Machine failure'] = np.where((data['TWF'] == 1) | (data['HDF'] == 1) | (data['PWF'] == 1) | (data['OSF'] == 1), 1, 0)
        data.drop(['UDI','Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)
        # add features
        subtraction_columns = make_difference_features(data.drop(columns=['Machine failure', 'Type']))
        multiplication_columns = make_multiplicative_features(data.drop(columns=['Machine failure', 'Type']))
        data = pd.concat([data, subtraction_columns, multiplication_columns], axis=1)
        # one-hot encode the quality type
        data = pd.get_dummies(data, columns=['Type',])

        # split and standardize the data
        X = data.drop(columns='Machine failure')
        y = data['Machine failure']
        self.X, _, self.y, _ = model_selection.train_test_split(X, y, test_size=self.testSize, random_state=self.randomSeed, stratify=y)

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)


    # params contains floats representing the following:
    # 'hidden_layer_sizes': up to 4 positive integers
    # 'activation': {'tanh', 'relu', 'logistic'},
    # 'alpha': float
    def convertParams(self, params):

        # transform the layer sizes from float (possibly negative) values into hiddenLayerSizes tuple:
        # if the the params[i] is negative, then that is the last layer and nothing beyond that is considered
        # one layer
        if round(params[1]) <= 0:
            hiddenLayerSizes = round(params[0]),
        # two layer
        elif round(params[2]) <= 0:
            hiddenLayerSizes = (round(params[0]), round(params[1]))
        # three layer
        else:
            hiddenLayerSizes = (round(params[0]), round(params[1]), round(params[2]))

        activation = ['tanh', 'relu', 'logistic'][floor(params[3])]
        alpha = params[4]

        return hiddenLayerSizes, activation, alpha

    @ignore_warnings(category=ConvergenceWarning)
    def getAccuracy(self, params):
        hiddenLayerSizes, activation, alpha = self.convertParams(params)

        self.classifier = MLPClassifier(random_state=self.randomSeed,
                                        hidden_layer_sizes=hiddenLayerSizes,
                                        activation=activation,
                                        solver='lbfgs',
                                        alpha=alpha,
                                        max_iter=300)

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy',
                                                     n_jobs=-1)

        return cv_results.mean()

    def formatParams(self, params):
        hiddenLayerSizes, activation, alpha = self.convertParams(params)
        return f"hidden_layer_sizes={hiddenLayerSizes}, activation='{activation}', solver='lbfgs', alpha={alpha}, max_iter=300"