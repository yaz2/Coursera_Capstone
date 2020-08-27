
# %% Imports
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
LabelName = 'SEVERITYCODE'


# %% Methods


def underSample(df):
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X = df.drop([LabelName], axis=1)
    y = df[LabelName]
    X_under, y_under = undersample.fit_resample(X, y)
    y_under = y_under.to_frame(name=LabelName)
    X_under[LabelName] = y_under

    return X_under


def buildFreqChart(df, colName):
    x = df[colName].unique()
    numValues = x.shape[0]
    arr1 = np.empty(numValues, dtype=float)
    arr2 = np.empty(numValues, dtype=float)
    totalArr = np.empty(numValues, dtype=float)
    index = 0
    numTotal = df.shape[0]
    xindex = 0
    for i in x:
        isNan = False
        try:
            if np.math.isnan(float(i)):
                isNan = True
                x[xindex] = '#NAN'
        except:
            isNan = False
        xindex = xindex + 1
        if isNan:
            df1 = df.loc[df[colName].isnull()]
        else:
            df1 = df.loc[df[colName] == i]
        numOfRows = df1.shape[0]

        dfc1 = df1.loc[df1[LabelName] == 1]
        dfc2 = df1.loc[df1[LabelName] == 2]
        assert numOfRows == (dfc1.shape[0] + dfc2.shape[0]), "Sum is not equal!!"
        try:
            arr1[index] = (dfc1.shape[0] / numOfRows) * 100
        except Exception as e:
            print(str(e) + ":" + colName + " with " + str(x))
            print(df.head())
            exit(0)
        arr2[index] = (dfc2.shape[0] / numOfRows) * 100
        totalArr[index] = (numOfRows / numTotal) * 100
        index = index + 1

    plt.figure(figsize=(2 + numValues, 3))

    # stack bars
    if df.dtypes[colName] == object:
        x = ['\n'.join(wrap(l, 12)) for l in x]

    plt.bar(x, arr1, label='Code 1', color='Green', width=0.3)
    plt.bar(x, arr2, bottom=arr1, label='Code 2', color='Red', width=0.3)
    plt.xticks(fontsize=8)

    # add text annotation corresponding to the percentage of each data.
    for xpos, ypos, yval in zip(x, arr1 / 2, arr1):
        plt.text(xpos, ypos, "%.1f" % yval, ha="center", va="center")
    for xpos, ypos, yval in zip(x, arr1 + arr2 / 2, arr2):
        plt.text(xpos, ypos, "%.1f" % yval, ha="center", va="center")

    # add text annotation corresponding to the "total" value of each bar
    for xpos, ypos, yval in zip(x, arr1 + arr2, totalArr):
        plt.text(xpos, ypos, "%.1f%s" % (yval, "%"), ha="center", va="bottom")

    plt.ylim(0, 110)
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
    plt.xlabel(colName, labelpad=5)
    plt.ylabel("Percentage", labelpad=5)
    plt.title("Severity Code Grouped by " + colName, y=1.02)
    plt.show()


def buildFreqChart1(df, colName):
    x = df[colName].unique()
    x.sort()
    numValues = x.shape[0]
    yArr = np.empty(numValues, dtype=float)
    index = 0
    for i in x:
        count = df.loc[df[colName] == i].shape[0]
        yArr[index] = count
        index = index + 1

    x = x.astype(str)
    plt.figure()
    plt.bar(x, yArr, width=0.2, align='center')
    for index, value in enumerate(yArr):
        plt.text(index - 0.04, value, f"{value:,.0f}")

    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.xlabel("Severity Code", labelpad=5)
    plt.ylabel("Count of Accidents", labelpad=5)
    plt.title("Accidents Grouped by Severity Code ", y=1.02)
    plt.show()
    # plt.savefig(FIGS_DIR + colName + str(random.randint(0,99))+ '.png', bbox_inches='tight', pad_inches=0.02)


def buildBoxPlot(df, colName):
    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(df[colName])
    plt.ylabel(colName, labelpad=5)
    plt.title("Distribution of " + colName, y=1.02)
    plt.show()


def fixDate(df):
    dateCol = 'INCDATE'
    timeCol = 'INCDTTM'
    time_format1 = '%I:%M:%S %p'
    time_format2 = '%H:%M:%S'
    df[dateCol] = pd.to_datetime(df[dateCol], format='%Y/%m/%d')
    invalidCount = 0
    for i in df.index:
        date = df[dateCol][i]
        time = df[timeCol][i]
        df.at[i, 'Month'] = date.strftime("%B")
        df.at[i, 'Day'] = date.strftime("%a")
        x = time.find(' ')
        timeStr = time[x + 1:]
        hour = -1
        try:
            hour = datetime.strptime(timeStr, time_format1).time().hour
        except ValueError:
            try:
                hour = datetime.strptime(timeStr, time_format2).time().hour
            except ValueError:
                invalidCount = invalidCount + 1
        df.at[i, 'Hour'] = hour

    print("Accidents with Invalid Time=" + str(invalidCount))
    df['Hour'] = df['Hour'].astype(str)

    return df


def computeCorCat(df, colName):
    table, results = rp.crosstab(df[colName], df['SEVERITYCODE'], prop='col', test='chi-square',
                                 correction=False)
    print("Correlation with " + colName)
    print(results)


