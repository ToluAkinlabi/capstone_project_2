#############################
# General utility functions
#############################

# 3rd party
import pandas as pd
from sklearn.metrics import accuracy_score

# standard library
from itertools import combinations

def make_multiplicative_features(df: pd.DataFrame, columns: list=[]) -> pd.DataFrame:
    R = 2
    if columns:
        column_combinations = combinations(columns, R)
    else:
        column_combinations = combinations(df.columns, R)

    new_features = {}
    for col_1, col_2 in column_combinations:
            new_features[f'{col_1} * {col_2}'] = df[col_1] * df[col_2]
    return pd.DataFrame(new_features)

def make_difference_features(df: pd.DataFrame, columns: list=[], absolute: bool=True) -> pd.DataFrame:
    R = 2
    if columns and absolute:
        column_combinations = combinations(columns, R)
    elif columns and not absolute:
        column_combinations = permutations(columns, R)
    elif not columns and absolute:
        column_combinations = combinations(df.columns, R)
    else:
        column_combinations = permutations(df.columns, R)

    new_features = {}
    for col_1, col_2 in column_combinations:
        if absolute:
            new_features[f'{col_1} - {col_2}'] = (df[col_1] - df[col_2]).abs()
        else:
            new_features[f'{col_1} - {col_2}'] = df[col_1] - df[col_2]
    return pd.DataFrame(new_features)

def calculate_and_display_accuracy(model, X_train, y_train, X_test, y_test, verbose: bool=True, rounding: int=4) -> list:
    model_train_accuracy = accuracy_score(y_train, model.predict(X_train))
    model_test_accuracy = accuracy_score(y_test, model.predict(X_test))
    baseline_train_accuracy = y_train.value_counts(normalize=True)[0]
    baseline_test_accuracy = y_test.value_counts(normalize=True)[0]
    if verbose:
        print(f'{"Model Training accuracy:":<58} {round(model_train_accuracy, rounding)}')
        print(f'{"Model Test accuracy:":<58} {round(model_test_accuracy, rounding)}')
        print(f'{"Baseline, Naive model (always guess 0) training accuracy:":<58} {round(baseline_train_accuracy, rounding)}')
        print(f'{"Baseline, Naive model (always guess 0) test accuracy:":<58} {round(baseline_test_accuracy, rounding)}')
    return model_train_accuracy, model_test_accuracy, baseline_train_accuracy, baseline_test_accuracy
