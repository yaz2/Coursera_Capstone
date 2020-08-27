#%% Imports
from pprint import pprint
import random

import matplotlib
import numpy as np
from textwrap import wrap
from datetime import date, datetime
import datetime as dt
from xgboost import XGBClassifier
from rfpimp import *
import pandas as pd
import researchpy as rp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from geopy.distance import geodesic
from imblearn.under_sampling import RandomUnderSampler
from geopy.distance import great_circle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

FIGS_DIR = 'figs/'
LabelName='SEVERITYCODE'


# %% Model Methods

def train_test(df):
    # Get the Dependent and Independent Features.
    X = df.drop([LabelName], axis=1)
    y = df[LabelName]

    # Split into 90% train and 10% test
    return train_test_split(X, y, test_size=0.10, shuffle=True, stratify=y)


def XGB(df):
    # Split to train test.. 90% <-> 10% (not shuffled)
    X_train, X_test, y_train, y_test = train_test(df)
    print("Features:\n" + str(df.dtypes))
    print("Start Training...")
    lr_list = [0.1]
    n_estimators = [128]
    max_depth = [3, 10, 2]
    subsample = [0.8]
    min_child_weight = [1, 6, 2]
    gamma = [0, 0.3]
    colsample_bytree = [0.8]
    scale_pos_weight = [1]

    search_grid = {
        'eta': lr_list,
        'n_estimators': n_estimators,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'colsample_bytree': colsample_bytree,
        'scale_pos_weight': scale_pos_weight,
        'max_depth': max_depth,
        'subsample': subsample}
    pprint(search_grid)

    rf = XGBClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=search_grid, cv=10, verbose=2,
                               n_jobs=-3, scoring='f1')

    grid_search.fit(X_train, y_train)

    pprint(grid_search.best_params_)
    print(grid_search.best_score_)

    best_grid = grid_search.best_estimator_
    res = best_grid.predict(X_test)
    print(confusion_matrix(y_test, res))
    print(f1_score(y_test, res))


def randomForest(df):
    # Split to train test.. 90% <-> 10% (not shuffled)
    X_train, X_test, y_train, y_test = train_test(df)
    print("Features:\n" + str(df.dtypes))
    print("Start Training...")
    # Parameters
    # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=64, stop=1024, num=10)]
    n_estimators = [64]

    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth = [40]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [65]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    # Create the random grid
    search_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(search_grid)

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=search_grid, cv=10, verbose=2,
                               n_jobs=-3, scoring='f1')

    grid_search.fit(X_train, y_train)

    pprint(grid_search.best_params_)
    print(grid_search.best_score_)

    best_grid = grid_search.best_estimator_
    res = best_grid.predict(X_test)
    cm = confusion_matrix(y_test, res)
    print(f1_score(y_test, res))
    print(accuracy_score(y_test, res))
    labels = ['Code 1', 'Code 2']
    print(cm)
    plot_confusion_matrix(best_grid, X_test, y_test)
    plt.title('Confusion Matrix of the Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(FIGS_DIR + 'cm' + '.png', pad_inches=50)

