##############################################################################
# Based on code from the 'Hands-On-Genetic-Algorithms-with-Python' repository
# https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter08
##############################################################################

# 3rd party
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv


class HyperparameterTuningGenetic:

    NUM_FOLDS = 3

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initMachineFailureDataset()
        self.kfold = model_selection.StratifiedKFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

    # MODIFY - to use the parsing code from the notebook
    def initMachineFailureDataset(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

        self.data = read_csv(url, header=None, usecols=range(0, 14))
        self.X = self.data.iloc[:, 1:14]
        self.y = self.data.iloc[:, 0]
    # MODIFY

    # MODIFY - use rf parameters
    # ADABoost [n_estimators, learning_rate, algorithm]:
    # "n_estimators": integer
    # "learning_rate": float
    # "algorithm": {'SAMME', 'SAMME.R'}
    def convertParams(self, params):
        n_estimators = round(params[0])  # round to nearest integer
        learning_rate = params[1]        # no conversion needed
        algorithm = ['SAMME', 'SAMME.R'][round(params[2])]  # round to 0 or 1, then use as index
        return n_estimators, learning_rate, algorithm
    # MODIFY

    # MODIFY - unpack into different params, use rf model
    def getAccuracy(self, params):
        n_estimators, learning_rate, algorithm = self.convertParams(params)
        self.classifier = AdaBoostClassifier(random_state=self.randomSeed,
                                             n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             algorithm=algorithm
                                             )

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')
        return cv_results.mean()
    # MODIFY

    # MODIFY - use f-string and different params
    def formatParams(self, params):
        return "'n_estimators'=%3d, 'learning_rate'=%1.3f, 'algorithm'=%s" % (self.convertParams(params))