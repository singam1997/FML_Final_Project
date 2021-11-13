import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import cross_val_score

#  PCA
from sklearn.decomposition import PCA

#plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def PCA_old_pro(file_name, components):
    df = pd.read_csv(file_name)
    # print("NULL Values",df.isnull().sum())
    # print("ISNULL",df.isnull().values.any())
    # print("TYPES",df.dtypes)
    # print(df.describe())
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
    # print(outlier_indices)
    # mask = []
    
    data_new = df[[i not in outlier_indices for i in range(0,df.shape[0])]]
    # print(type(data_new))
    df = data_new

    train, test = train_test_split(df, test_size=0.2)

    # print(type(train),type(test))

    train_Y = train["Outcome"].to_frame()
    train_X = train.drop("Outcome",axis=1)

    test_Y = test["Outcome"].to_frame()
    test_X = test.drop("Outcome",axis=1)

    train_X_np = train_X.values
    test_X_np = test_X.values

    
    pca = PCA(n_components=components)
    
    train_X_np = pca.fit(train_X_np).transform(train_X_np)
    test_X_np = pca.transform(test_X_np)

    return train_X_np, train_Y.values.flatten(), test_X_np, test_Y.values.flatten()


def PCA_pre_pro(file_name, components):
    df = pd.read_csv(file_name)
    train, test = train_test_split(df, test_size=0.2)
    
    train_Y = train["Outcome"].to_frame()
    train_X = train.drop("Outcome",axis=1)
    
    test_Y = test["Outcome"].to_frame()
    test_X = test.drop("Outcome",axis=1)

    train_X_np = train_X.values
    test_X_np = test_X.values

    
    pca = PCA(n_components=components)
    
    train_X_np = pca.fit(train_X_np).transform(train_X_np)
    test_X_np = pca.transform(test_X_np)

    return train_X_np, train_Y.values.flatten(), test_X_np, test_Y.values.flatten()

def plot_PCA_data(file_name):
    df = pd.read_csv(file_name)
    
    
    train_Y = df["Outcome"].to_frame()
    train_X = df.drop("Outcome",axis=1)
    # train_X = preprocessing.scale(train_X)

    pca = PCA(n_components=3)
    train_X_np = np.array(train_X)
    train_X_np = (train_X_np-train_X_np.mean())/train_X_np.std()
    train_X_np = pca.fit_transform(train_X_np)
    z_vals = train_X_np[np.array(train_Y==0).flatten()]
    y_vals = train_X_np[np.array(train_Y==1).flatten()]
    
    z_x_vals = list(z_vals[:,0])
    z_y_vals = list(z_vals[:,1])
    z_z_vals = list(z_vals[:,2])

    y_x_vals = list(y_vals[:,0])
    y_y_vals = list(y_vals[:,1])
    y_z_vals = list(y_vals[:,2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(z_x_vals, z_y_vals, z_z_vals, c='r', marker='o')
    ax.scatter(y_x_vals, y_y_vals, y_z_vals, c='b', marker='^')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def no_pre_processing(file_name):
	df = pd.read_csv(file_name)
    
	train, test = train_test_split(df, test_size=0.2)

    # print(type(train),type(test))

	train_Y = train["Outcome"].to_frame()
	train_X = train.drop("Outcome",axis=1)

	test_Y = test["Outcome"].to_frame()
	test_X = test.drop("Outcome",axis=1)

	return train_X.values, train_Y.values.flatten(), test_X.values, test_Y.values.flatten()

def read_data(file_name):
    df = pd.read_csv(file_name)
    # print("NULL Values",df.isnull().sum())
    # print("ISNULL",df.isnull().values.any())
    # print("TYPES",df.dtypes)
    # print(df.describe())
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
    # print(outlier_indices)
    # mask = []
    
    data_new = df[[i not in outlier_indices for i in range(0,df.shape[0])]]
    # print(type(data_new))
    df = data_new

    train, test = train_test_split(df, test_size=0.2)

    # print(type(train),type(test))

    train_Y = train["Outcome"].to_frame()
    train_X = train.drop("Outcome",axis=1)

    test_Y = test["Outcome"].to_frame()
    test_X = test.drop("Outcome",axis=1)

    return train_X.values, train_Y.values.flatten(), test_X.values, test_Y.values.flatten()

def train_data(train_X, train_Y, test_X, test_Y):
    # print()
    # print()
    # print()
    # print()
    gnb = GaussianNB()
    y_pred = gnb.fit(train_X, train_Y).predict(test_X)
    # print("ACCURACY=",(test_Y==y_pred).sum()/test_Y.shape[0])
    # print("Accuracy=", gnb.score(test_X,test_Y))
    
    print(gnb.score(test_X,test_Y))

    # print("balanced_accuracy_score:",balanced_accuracy_score(test_Y, y_pred))
    # print("RECALL:",recall_score(test_Y, y_pred))
    # print("precision_recall_fscore_support:",precision_recall_fscore_support(test_Y, y_pred))
    # print("multilabel_confusion_matrix",multilabel_confusion_matrix(test_Y, y_pred))
    # scores = cross_val_score(gnb, np.array(list(train_X)+list(test_X)), np.array(list(train_Y)+list(test_Y)), cv=5)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# train_X, train_Y, test_X, test_Y = read_data("diabetes.csv")
# train_data(train_X, train_Y, test_X, test_Y)

# train_X, train_Y, test_X, test_Y = no_pre_processing("diabetes.csv")
# train_data(train_X, train_Y, test_X, test_Y)

# train_X, train_Y, test_X, test_Y = PCA_pre_pro("diabetes.csv",int(sys.argv[1]))
# train_data(train_X, train_Y, test_X, test_Y)

# PCA_old_pro
# train_X, train_Y, test_X, test_Y = PCA_old_pro("diabetes.csv",int(sys.argv[1]))
# train_data(train_X, train_Y, test_X, test_Y)

plot_PCA_data("diabetes.csv")