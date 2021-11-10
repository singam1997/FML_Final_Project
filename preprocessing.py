import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score


def read_data(file_name):
    df = pd.read_csv(file_name)
    print("NULL Values",df.isnull().sum())
    print("ISNULL",df.isnull().values.any())
    print("TYPES",df.dtypes)
    print(df.describe())
    col_names = list(df.columns)
    outlier_indices = []
    for i in col_names:
        Q1 = np.quantile(df[i],0.25)
        Q3 = np.quantile(df[i],0.75)
        IQR = Q3 - Q1

        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        # print(lower_range, upper_range)
        for j in range(0,(df.shape[0])):
            if (df[i][j] < lower_range) or (df[i][j] > upper_range):
                # print("out_ind",outlier_indices)        
                outlier_indices.append(j)
        
        if outlier_indices!=[]:
            outlier_indices = list(set(outlier_indices))
    outlier_indices.sort()
    print(outlier_indices)
    # mask = []
    
    data_new = df[[i not in outlier_indices for i in range(0,df.shape[0])]]
    print(type(data_new))
    
    train, test = train_test_split(df, test_size=0.2)

    # print(type(train),type(test))

    train_Y = train["Outcome"].to_frame()
    train_X = train.drop("Outcome",axis=1)

    test_Y = test["Outcome"].to_frame()
    test_X = test.drop("Outcome",axis=1)

    return train_X.values, train_Y.values.flatten(), test_X.values, test_Y.values.flatten()

def train_data(train_X, train_Y, test_X, test_Y):
    print()
    print()
    print()
    print()
    gnb = GaussianNB()
    y_pred = gnb.fit(train_X, train_Y).predict(test_X)
    print("ACCURACY=",(test_Y==y_pred).sum()/test_Y.shape[0])
    print("balanced_accuracy_score:",balanced_accuracy_score(test_Y, y_pred))
    print("RECALL:",recall_score(test_Y, y_pred))
    print("precision_recall_fscore_support:",precision_recall_fscore_support(test_Y, y_pred))
    print("multilabel_confusion_matrix",multilabel_confusion_matrix(test_Y, y_pred))

train_X, train_Y, test_X, test_Y = read_data("diabetes.csv")
train_data(train_X, train_Y, test_X, test_Y)