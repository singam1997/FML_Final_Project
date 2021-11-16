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

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier

def plot_X_Y(X,Y, name):

    X_new = list(X)
    Y_new = list(Y)

    plt.plot(X_new, Y_new, label = name)
    
    plt.xlabel('x - axis')
    
    plt.ylabel('y - axis')
    
    plt.title(name)
     
    plt.legend()

    plt.show()


def ada_boost(X,Y,estimators):
    gnb = GaussianNB()
    clf = AdaBoostClassifier(base_estimator=GaussianNB(),n_estimators=estimators, random_state=0)
    # clf.fit(X,Y)
    clf.fit(np.array(X),np.array(Y).flatten())
    # print("Ada Acc:",clf.score(np.array(X),np.array(Y).flatten()))
    return clf.score(np.array(X),np.array(Y).flatten())

def see_LDA(X,Y):
    ##PLOT DATA AFTER LDA TRANSFORMATION
    clf = LinearDiscriminantAnalysis()
    clf.fit(np.array(X), np.array(Y).flatten())
    
    one_x = clf.transform(np.array(X)).flatten()[np.array(Y).flatten()==1]
    zero_x = clf.transform(np.array(X)).flatten()[np.array(Y).flatten()==0]

    plt.plot(list(zero_x),np.zeros_like(zero_x)+1,'x')
    plt.plot(list(one_x),np.zeros_like(one_x)+2,'^')
    plt.show()


def PCA_train(train_X,train_Y,test_X,test_Y, components):
    ##TRANSFORM TO A LOW DIMENSIONAL SUBSPACE USING PCA
    pca = PCA(n_components=components)

    train_X = pca.fit(train_X).transform(train_X)
    test_X = pca.transform(test_X)

    return train_X, train_Y, test_X, test_Y

def plot_PCA_data(X,Y):
    ##PLOT 3D PCA WITH REAL X VALUES AND BOOLEAN Y VALUES
    pca = PCA(n_components=3)
    X = (X-X.mean())/X.std()
    X = pca.fit_transform(X)

    z_vals = X[np.array(Y==0).flatten()]
    y_vals = X[np.array(Y==1).flatten()]
    
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

def split_data(X, Y):
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    df = pd.concat([df_x,df_y],axis=1)
    
    train, test = train_test_split(df, test_size=0.2)
    print()
    
    train_Y = np.array(train)[:,-1]
    train_X = np.array(train)[:,:-1]

    test_Y = np.array(test)[:,-1]
    test_X = np.array(test)[:,:-1]

    return train_X, train_Y, test_X, test_Y

def read_raw_data(file_name):
    df = pd.read_csv(file_name)

    # print("NULL Values",df.isnull().sum())
    # print("ISNULL",df.isnull().values.any())
    # print("TYPES",df.dtypes)
    # print(df.describe())
    # col_names = list(df.columns)
    
    Y = df["Outcome"].to_frame()
    X = df.drop("Outcome",axis=1)

    return X.values, Y.values.flatten()

def rem_outliers(X,Y):
    cols = X.shape[1]
    mask = np.array([False]*X.shape[0])
    # outlier_indices = []
    for i in range(0,X.shape[1]):
        # print(X[:,i])
        Q1 = np.quantile(X[:,i],0.25)
        Q3 = np.quantile(X[:,i],0.75)
        IQR = Q3 - Q1

        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        
        mask = np.logical_or(X[:,i]<lower_range, mask)
        mask = np.logical_or(X[:,i]>upper_range, mask)
        # print(mask.sum())
        # print(mask)
    X_new = X[mask==False]
    Y_new = Y[mask==False]
    return X_new,Y_new

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


if __name__=="__main__":
    X,Y = read_raw_data("diabetes.csv")
    print(X.shape)
    print(Y.shape)
    X,Y = rem_outliers(X,Y)
    print(X.shape)
    print(Y.shape)
    train_X, train_Y, test_X, test_Y = split_data(X,Y)
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    # train_data(train_X, train_Y, test_X, test_Y)
    # plot_PCA_data(train_X, train_Y)
    # plot_PCA_data(test_X, test_Y)
    
    # train_X, train_Y, test_X, test_Y = PCA_train(train_X, train_Y, test_X, test_Y, 3)
    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    # see_LDA(train_X, train_Y)
    # see_LDA(test_X, test_Y)
    plot_X = []
    plot_Y = []
    for i in range(1,100):
        # print(i,end="--")
        # ada_boost(train_X,train_Y,i)
        plot_X.append(i)
        plot_Y.append(ada_boost(train_X, train_Y,i))
    plot_X_Y(plot_X, plot_Y, "PLOT")
# train_X, train_Y, test_X, test_Y = rem_outliers("diabetes.csv")
# train_data(train_X, train_Y, test_X, test_Y)

# train_X, train_Y, test_X, test_Y = read_raw_data("diabetes.csv")
# train_data(train_X, train_Y, test_X, test_Y)

# train_X, train_Y, test_X, test_Y = PCA_train("diabetes.csv",int(sys.argv[1]))
# train_data(train_X, train_Y, test_X, test_Y)

# PCA_old_pro
# train_X, train_Y, test_X, test_Y = PCA_old_pro("diabetes.csv",int(sys.argv[1]))
# train_data(train_X, train_Y, test_X, test_Y)

# plot_PCA_data("diabetes.csv")

# see_LDA("diabetes.csv")
# ada_boost("diabetes.csv")
