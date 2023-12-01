##############################################################################
# Based on code from the 'Hands-On-Genetic-Algorithms-with-Python' repository
# https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter08
##############################################################################

# 3rd party
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# internal imports
from util import make_difference_features, make_multiplicative_features

# constants
RANDOM_SEED = 42
TEST_SIZE = 0.25


class HyperparameterTuningGenetic:

    NUM_FOLDS = 3

    def __init__(self, randomSeed: int=RANDOM_SEED):
        
        self.randomSeed = randomSeed
        self.initMachineFailureDataset()
        self.kfold = model_selection.StratifiedKFold(n_splits=self.NUM_FOLDS) #, random_state=self.randomSeed)

    def initMachineFailureDataset(self):
        data = pd.read_csv('../machine failure.csv')

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
        self.X, _, self.y, _ = model_selection.train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

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
                                                     scoring='accuracy')
        return cv_results.mean()

    def formatParams(self, params):
        samples, features, depth, estimators = self.convertParams(params)
        return f'min_samples_leaf={samples}, max_features={features}, max_depth={depth}, estimators={estimators}'