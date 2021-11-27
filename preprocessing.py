import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
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
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated



# separated = dict()
# separated[0] = list()
# separated[1] = list()

# list(map(lambda dp: separated[dp[1]].append(dp[0]), list(zip(x, y))))

# class0 = np.vstack(separated[0])
# class1 = np.vstack(separated[0])

# mean_class0 = np.mean(class0, axis=0)
# std_class0 = np.std(class0, axis=0)

# mean_class1 = np.mean(class1, axis=0)
# std_class1 = np.std(class1, axis=0)





# for i in range(0,len(train_Y)):
#     if i==0:
#         separated[0].append()



def plot_X_Y(X,Y, name):

    X_new = list(X)
    Y_new = list(Y)

    plt.plot(X_new, Y_new, label = name)
    
    plt.xlabel('x - axis')
    
    plt.ylabel('y - axis')
    
    plt.title(name)
     
    plt.legend()

    plt.show()



def ada_boost_test_acc(train_x, train_y, test_x, test_y, estimators):
    uniq = [round(max(np.concatenate((train_x,test_x))[:,i])+1) for i in range(0,6)]
    adab = AdaBoostClassifier(base_estimator=CategoricalNB(min_categories=uniq),n_estimators=estimators, random_state=0)
    # clf.fit(X,Y)
    adab.fit(np.array(train_x),np.array(train_y).flatten())
    # print("Ada Acc:",clf.score(np.array(X),np.array(Y).flatten()))
    return adab.score(np.array(test_x),np.array(test_y).flatten())
    

def ada_boost_test_acc_gnb(train_x, train_y, test_x, test_y, estimators):
    # uniq = [round(max(np.concatenate((train_x,test_x))[:,i])+1) for i in range(0,6)]
    adab = AdaBoostClassifier(base_estimator=GaussianNB(),n_estimators=estimators, random_state=0)
    # clf.fit(X,Y)
    adab.fit(np.array(train_x),np.array(train_y).flatten())
    # print("Ada Acc:",clf.score(np.array(X),np.array(Y).flatten()))
    return adab.score(np.array(test_x),np.array(test_y).flatten())
    

def ada_boost(X,Y,estimators):
    # gnb = GaussianNB()
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
    # random_n = np.random.randint()
    # print("Random_State",random_n)
    train, test = train_test_split(df, test_size=0.2)#,random_state=random_n)
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
    
    for i in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        X[i] = X[i].replace(0,np.nan).fillna(value=round(X[i].mean()))

    return X.values, Y.values.flatten()

def feature_scaling(X,Y,params=[],num=0):
    
    if params==[]:
        mean = np.mean(X,axis=0)
        std = np.std(X,axis=0)
        min_val = np.min(X,axis=0)
        max_val = np.max(X,axis=0)
    
    else:
        num = params[0]

        mean = params[1]
        std = params[2]
        
        min_val = params[1]
        max_val = params[2]

    if num==-1:
        ##just return without feature scaling
        X_new = X
        params = [num,mean, std]

    if num==0:
        ###standardize
        X_new = (X-mean)/std
        params = [num,mean,std]

    if num==1:
        ##normalize
        X_new = (X-min_val)/(max_val-min_val)
        params = [num,min_val,max_val]

    if num==2:
        ###origin shifting
        X_new = (X-min_val)
        params = [num,min_val,std]

    if num==3:
        # print(X.shape)
        X_new = X[:,[0,1,2,3,4,7]]
        params = [num,mean, std]
        ###remove BMI and Diabetes data

    if num==4:
        print(num)
        print(X.shape)
        X_new = np.round(X)
        params = [num,mean, std]

    return X_new, Y, params    

def get_outlier_matrix(X,Y):
    cols = X.shape[1]
    outlier_matrix = np.array([[False]*X.shape[1]]*X.shape[0])
    for i in range(0,X.shape[1]):
        
        Q1 = np.quantile(X[:,i],0.25)
        Q3 = np.quantile(X[:,i],0.75)
        IQR = Q3 - Q1
        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        outlier_matrix[:,i] = np.logical_or(X[:,i]<lower_range,outlier_matrix[:,i])
        outlier_matrix[:,i] = np.logical_or(X[:,i]>upper_range,outlier_matrix[:,i])
    return outlier_matrix

def rem_outliers(X,Y):
    cols = X.shape[1]
    mask = np.array([False]*X.shape[0])
    # outlier_indices = []
    for i in range(0,X.shape[1]):
        # print(X[:,i])
        # print(i)
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

    X_outliers = X[mask==True]
    Y_outliers = Y[mask==True]
    return X_new,Y_new, X_outliers, Y_outliers

def get_outlier_weights(train_Y, outlier_column, outlier_weight=-1, zero_weight=-1, one_weight=-1):
    z_examples = ((train_Y)==0).sum()
    o_examples = ((train_Y)==1).sum()
    weight_vector = np.zeros(train_Y.flatten().shape[0])
    
    if zero_weight==-1:
        zero_weight = 1+o_examples/(o_examples+z_examples)

    if one_weight==-1:
        one_weight = 1+2.4*z_examples/(o_examples+z_examples)
    
    if outlier_weight==-1:
        outlier_weight=0.3
    
    weight_vector[train_Y==0] = zero_weight
    weight_vector[train_Y==1] = one_weight
    weight_vector[outlier_column==True] = outlier_weight
    print("Weight_vector",weight_vector)
    return weight_vector


def get_weights(train_X, train_Y):
    z_examples = ((train_Y)==0).sum()
    o_examples = ((train_Y)==1).sum()
    
    z_weights = o_examples
    o_weights = z_examples
    # print("zexamples, o_examples",z_weights, o_weights)
    z_weights = 3#round(z_examples/100)
    o_weights = 5#round(o_examples/100)

    # z_weights = 1+z_weights/(o_examples+z_examples)
    # o_weights = 1+o_weights/(o_examples+z_examples)
    # print(z_weights, o_weights)
    get_weights = np.zeros(train_Y.size)
    # print(get_weights.shape, get_weights[train_Y==0])
    get_weights[train_Y==0]=z_weights
    get_weights[train_Y==1]=o_weights
    # print(get_weights)
    return get_weights



def awesome_mixture_nb(train_X, train_Y, test_X, test_Y, categories,outliers_matrix = np.array([])):
    ##if categories are 1 predict prob using gaussian distribution
    ##if categories are 0 predict prop using categorical
    models = []
    ###training
    for i in range(0,train_X.shape[1]):
        
        train_feature = train_X[:,i].reshape(-1,1)
        
        if categories[i]==0:
            train_feature = np.round(train_feature)

        if outliers_matrix.size!=0:
            feature_weights = get_outlier_weights(train_Y, outliers_matrix[:,i])
        else:
            feature_weights = None

        if categories[i]==0:
            models.append(MultinomialNB())
            models[i].fit(train_feature, train_Y, feature_weights)

        if categories[i]==1:
            models.append(GaussianNB())
            models[i].fit(train_feature, train_Y, feature_weights)

    ##testing
    log_probs = np.zeros((test_X.shape[0],2))
    for i in range(0,test_X.shape[1]):
        
        test_feature = test_X[:,i].reshape(-1,1)
        
        if categories[i]==0:
            test_feature = np.round(test_feature)

        log_probs += models[i].predict_log_proba(test_feature)

    # x_categorical = train_X[:,np.array(categories)==0]
    # x_categorical = np.round(x_categorical)


    # x_gaussian = train_X[:,np.array(categories)==1]


    # gnb = GaussianNB()
    # gnb.fit(x_gaussian, train_Y,weights)

    # clf = MultinomialNB()#min_categories=uniq)
    # clf.fit(x_categorical, train_Y,weights)

    # test_x_categorical = test_X[:,np.array(categories)==0]
    # test_x_categorical = np.round(test_x_categorical)
    # test_x_gaussian = test_X[:,np.array(categories)==1]

    # test_x_log_prob_categorical = clf.predict_log_proba(test_x_categorical)
    # test_x_log_prob_gaussian = gnb.predict_log_proba(test_x_gaussian)

    # log_sum = test_x_log_prob_categorical + test_x_log_prob_gaussian
    prediction = []
    # print(log_probs)
    # exit()
    for i in log_probs:
        if i[0] < i[1]:
            prediction.append(1)
        else:
            prediction.append(0)

    y_pred = np.array(prediction)
    print("---------------------------")
    print("Awesome_Mix_acc:",(((test_Y==y_pred).sum())/test_Y.shape[0]))
    # print("balanced_accuracy_score:",balanced_accuracy_score(test_Y, y_pred))
    print("Awesome_PRECISION:",precision_score(test_Y, y_pred))
    print("Awesome_RECALL:",recall_score(test_Y, y_pred))
    # print("precision_recall_fscore_support:",precision_recall_fscore_support(test_Y, y_pred))
    # print("multilabel_confusion_matrix",multilabel_confusion_matrix(test_Y, y_pred))
    print("--------------------------")


def mixture_nb(train_X, train_Y, test_X, test_Y, categories,outliers_X=np.array([]), outliers_Y=np.array([])):
    ##if categories are 1 predict prob using gaussian distribution
    ##if categories are 0 predict prop using categorical
    x_categorical = train_X[:,np.array(categories)==0]
    x_categorical = np.round(x_categorical)

    # x_outlier_categorical = np.round(outliers_X[:,np.array(categories)==0])


    x_gaussian = train_X[:,np.array(categories)==1]

    # uniq = [round(max(np.concatenate((outliers_X,train_X,test_X))[:,i])+1) for i in range(0,train_X.shape[1])]
    # uniq = list(np.array(uniq)[np.array(categories)==0])
    weights = get_weights(train_X, train_Y)
    

    gnb = GaussianNB()
    gnb.fit(x_gaussian, train_Y,weights)

    clf = MultinomialNB()#min_categories=uniq)
    clf.fit(x_categorical, train_Y,weights)

    test_x_categorical = test_X[:,np.array(categories)==0]
    test_x_categorical = np.round(test_x_categorical)
    test_x_gaussian = test_X[:,np.array(categories)==1]

    test_x_log_prob_categorical = clf.predict_log_proba(test_x_categorical)
    test_x_log_prob_gaussian = gnb.predict_log_proba(test_x_gaussian)

    log_sum = test_x_log_prob_categorical + test_x_log_prob_gaussian
    prediction = []

    for i in log_sum:
        if i[0] < i[1]:
            prediction.append(1)
        else:
            prediction.append(0)

    y_pred = np.array(prediction)
    print("---------------------------")
    print("Mix_acc:",(((test_Y==y_pred).sum())/test_Y.shape[0]))
    # print("balanced_accuracy_score:",balanced_accuracy_score(test_Y, y_pred))
    print("PRECISION:",precision_score(test_Y, y_pred))
    print("RECALL:",recall_score(test_Y, y_pred))
    # print("precision_recall_fscore_support:",precision_recall_fscore_support(test_Y, y_pred))
    # print("multilabel_confusion_matrix",multilabel_confusion_matrix(test_Y, y_pred))
    print("--------------------------")

def train_categoNB(train_X, train_Y, test_X, test_Y):
    # uniq = [np.unique(np.concatenate((train_X,test_X))[:,i]).size for i in range(0,6)]
    uniq = [round(max(np.concatenate((train_X,test_X))[:,i])+1) for i in range(0,6)]
    # print(uniq)
    clf = CategoricalNB(min_categories=uniq)
    weights = get_weights(train_X, train_Y)
    clf.fit(train_X, train_Y,weights)
    # print(clf.n_categories_)
    # print([max(np.concatenate((train_X,test_X))[:,i])+1 for i in range(0,6)])
    y_pred = clf.predict(test_X)
    # print("ACCURACY=",(test_Y==y_pred).sum()/test_Y.shape[0])
    print("CLF Accuracy=", clf.score(test_X,test_Y))

def train_ComplementNB(train_X, train_Y, test_X, test_Y):
    clf = ComplementNB()
    weights = get_weights(train_X, train_Y)
    clf.fit(train_X, train_Y,weights)
    y_pred = clf.predict(test_X)
    print("ComplementNB Accuracy=", clf.score(test_X,test_Y))

def train_multi_nb(train_X, train_Y, test_X, test_Y):
    clf = MultinomialNB()
    weights = get_weights(train_X, train_Y)
    clf.fit(train_X, train_Y,weights)
    y_pred = clf.predict(test_X)
    print("MultinomialNB Accuracy=", clf.score(test_X,test_Y))

def train_data(train_X, train_Y, test_X, test_Y):

    gnb = GaussianNB()
    weights = get_weights(train_X, train_Y)
    y_pred = gnb.fit(train_X, train_Y,weights).predict(test_X)
    # print("ACCURACY=",(test_Y==y_pred).sum()/test_Y.shape[0])
    print("GNB Accuracy=", gnb.score(test_X,test_Y))



    # print(gnb.score(test_X,test_Y))

    # print("balanced_accuracy_score:",balanced_accuracy_score(test_Y, y_pred))
    # print("RECALL:",recall_score(test_Y, y_pred))
    # print("precision_recall_fscore_support:",precision_recall_fscore_support(test_Y, y_pred))
    # print("multilabel_confusion_matrix",multilabel_confusion_matrix(test_Y, y_pred))
    # scores = cross_val_score(gnb, np.array(list(train_X)+list(test_X)), np.array(list(train_Y)+list(test_Y)), cv=5)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))




"""PASTE INTO TERMINAL

import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB

df = pd.read_csv("diabetes.csv")

Y = df["Outcome"].to_frame()
X = df.drop("Outcome",axis=1)
for i in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        X[i] = X[i].replace(0,np.nan).fillna(value=round(X[i].mean()))


x = X.values
# x = x[:,[0,1,2,3,4,7]]
y = Y.values.flatten()

X = x
Y = y
df_x = pd.DataFrame(X)
df_y = pd.DataFrame(Y)
df = pd.concat([df_x,df_y],axis=1)

train, test = train_test_split(df, test_size=0.2)
print()

train_Y = np.array(train)[:,-1]
train_X = np.array(train)[:,:-1]

test_Y = np.array(test)[:,-1]
test_X = np.array(test)[:,:-1]



categories = [0,0,0,0,0,1,1,0]
"""


if __name__=="__main__":
    X,Y = read_raw_data("diabetes.csv")
    print("Before removing outliers shape:", X.shape,Y.shape)
        
    
    
    train_X, train_Y, test_X, test_Y = split_data(X,Y)
    print("Training and testing shape:",train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    
    # train_X,train_Y,outliers_X, outliers_Y = rem_outliers(train_X,train_Y)
    print("After removing outliers shape:", train_X.shape,train_Y.shape)
    
    outlier_matrix = get_outlier_matrix(train_X, train_Y)
    awesome_mixture_nb(train_X, train_Y, test_X, test_Y, [0,1,0,0,1,1,1,0], outlier_matrix)
    # train_data(train_X, train_Y, test_X, test_Y)
    # mixture_nb(train_X, train_Y, test_X, test_Y, [0,1,0,0,1,1,1,0])
    
    train_X, train_Y, params = feature_scaling(train_X, train_Y,[],3)
    test_X, test_Y, params = feature_scaling(test_X, test_Y,params,3)
    print(train_X.shape, test_X.shape)
    
    train_categoNB(train_X, train_Y, test_X, test_Y)
    train_multi_nb(train_X, train_Y, test_X, test_Y)
    train_ComplementNB(train_X, train_Y, test_X, test_Y)
    # plot_PCA_data(train_X, train_Y)
    # plot_PCA_data(test_X, test_Y)
    
    # PCA_COMP = 3
    # train_X, train_Y, test_X, test_Y = PCA_train(train_X, train_Y, test_X, test_Y, PCA_COMP)
    # print("AFTER PCA:")
    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    # train_data(train_X, train_Y, test_X, test_Y)
    
    # see_LDA(train_X, train_Y)
    # see_LDA(test_X, test_Y)
    
#uncomment for adaboost
    # plot_X = []
    # plot_Y = []
    # for i in range(1,100):
    #     # print(i,end="--")
    #     # ada_boost(train_X,train_Y,i)
    #     plot_X.append(i)
    #     plot_Y.append(ada_boost_test_acc_gnb(train_X, train_Y,test_X,test_Y,i))
    

    # all_indices = []
    # for i in range(0,len(plot_Y)):
    #     if plot_Y[i]==max(plot_Y):
    #         all_indices.append(i)

    # print("Max AdaBoost Acc:",max(plot_Y), list(np.array(plot_X)[all_indices]))
    # plot_X_Y(plot_X, plot_Y, "PLOT")






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
