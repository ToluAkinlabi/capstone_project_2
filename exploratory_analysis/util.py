#############################
# General utility functions
#############################

# 3rd party
import pandas as pd

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

