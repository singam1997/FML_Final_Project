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

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import f1_score

import itertools
from itertools import combinations, chain
import warnings
from sklearn.exceptions import UndefinedMetricWarning
def findsubsets(s, n):
    return list(map(set, itertools.combinations(s, n)))

def feature_scaling(X,Y,params=[],num=0,abelation_set=[]):
    
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


    if num==5:
        X_new = X[:,np.array(abelation_set)==1]



    return X_new, Y, params

def split_data(X, Y):
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    df = pd.concat([df_x,df_y],axis=1)
    # random_n = np.random.randint()
    # print("Random_State",random_n)
    train, test = train_test_split(df, test_size=0.2)#,random_state=random_n)
    # print()
    
    train_Y = np.array(train)[:,-1]
    train_X = np.array(train)[:,:-1]

    test_Y = np.array(test)[:,-1]
    test_X = np.array(test)[:,:-1]

    return train_X, train_Y, test_X, test_Y


def read_raw_data(file_name):
    df = pd.read_csv(file_name)    
    Y = df["Outcome"].to_frame()
    X = df.drop("Outcome",axis=1)
    
    for i in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        X[i] = X[i].replace(0,np.nan).fillna(value=(X[i].mean()))
    means = dict()
    means['Glucose']=X['Glucose'].mean()
    means['BloodPressure']=X['BloodPressure'].mean()
    means['SkinThickness']=X['SkinThickness'].mean()
    means['Insulin']=X['Insulin'].mean()
    means['BMI']=X['BMI'].mean()

    return X.values, Y.values.flatten(), means



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


def get_outlier_weights(train_Y, outlier_column, outlier_weight=-1, zero_weight=-1, one_weight=-1):
    z_examples = ((train_Y)==0).sum()
    o_examples = ((train_Y)==1).sum()
    weight_vector = np.zeros(train_Y.flatten().shape[0])
    
    if zero_weight==-1:
        zero_weight = 1+o_examples/(o_examples+z_examples)

    if one_weight==-1:
        one_weight = 1+2.26*z_examples/(o_examples+z_examples)
    
    if outlier_weight==-1:
        outlier_weight=3
    
    weight_vector[train_Y==0] = zero_weight
    weight_vector[train_Y==1] = one_weight
    weight_vector[outlier_column==True] = outlier_weight
    # print("Weight_vector",weight_vector)
    return weight_vector

def calc_score(models, test_X, test_Y, categories):
    log_probs = np.zeros((test_X.shape[0],2))
    for i in range(0,test_X.shape[1]):
        
        test_feature = test_X[:,i].reshape(-1,1)
        
        if categories[i]==0:
            test_feature = np.round(test_feature)

        log_probs += models[i].predict_log_proba(test_feature)

    prediction = []

    for i in log_probs:
        if i[0] < i[1]:
            prediction.append(1)
        else:
            prediction.append(0)

    y_pred = np.array(prediction)
    # print("---------------------------")
    # print("Awesome_Mix_acc:",(((test_Y==y_pred).sum())/test_Y.shape[0]))
    # print("balanced_accuracy_score:",balanced_accuracy_score(test_Y, y_pred))
    exc_occr = 0
    try:
    # with warnings.catch_warnings(): 
        warnings.simplefilter("error")
        # warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        # warnings.filterwarnings('never')  
        # print("HHHHHHHHHHHHHHHHHHHHHHHHH")
        Precision = precision_score(precision_score(test_Y, y_pred))
        Recall = recall_score(test_Y, y_pred)
        F1 = f1_score(test_Y, y_pred, average=None)
        # print("precision_recall_fscore_support:",precision_recall_fscore_support(test_Y, y_pred))
        # cf_matrix = confusion_matrix(test_Y, y_pred)

        # print(classification_report(test_Y,  y_pred))


        # sns.heatmap(cf_matrix, annot=True, fmt='g')
        # print("multilabel_confusion_matrix",multilabel_confusion_matrix(test_Y, y_pred))
        # print("--------------------------")
    except:
        # print("An exception occurred")
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        
        exc_occr=1
        Precision = precision_score(test_Y, y_pred)
        Recall = recall_score(test_Y, y_pred)
        F1 = f1_score(test_Y, y_pred, average=None)
        # print("FROM exception")
        # print("precision_recall_fscore_support:",precision_recall_fscore_support(test_Y, y_pred))
        
        # cf_matrix = confusion_matrix(test_Y, y_pred)

        # print(classification_report(test_Y,  y_pred))


        # sns.heatmap(cf_matrix, annot=True, fmt='g')
        # print("multilabel_confusion_matrix",multilabel_confusion_matrix(test_Y, y_pred))
        # print("--------------------------")
    return Precision, Recall, F1, (((test_Y==y_pred).sum())/test_Y.shape[0])
    # print("Awesome_RECALL:",recall_score(test_Y, y_pred))
    # print("Awesome_abelation",f1_score(test_Y, y_pred, average=None))
        
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
            # print("I",i)
            models.append(MultinomialNB())
            models[i].fit(train_feature, train_Y, feature_weights)

        if categories[i]==1:
            # print("I",i)
            models.append(GaussianNB())
            models[i].fit(train_feature, train_Y, feature_weights)

    ##testing

    
    # print("MODEL SIZE", len(models),len(categories),train_X.shape[1],categories)
    return models





if __name__=="__main__":

    df_tmp = pd.read_csv("diabetes.csv")
    
    X,Y,means = read_raw_data("diabetes.csv")
    
    train_X, train_Y, test_X, test_Y = split_data(X,Y)
    # print("Training and testing shape:",train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    
    # train_X,train_Y,outliers_X, outliers_Y = rem_outliers(train_X,train_Y)
    # print("After removing outliers shape:", train_X.shape,train_Y.shape)
    K=1
    N=8
    all_combinations = [np.reshape(np.array(i), (K, N)) for i in itertools.product([0, 1], repeat = K*N)]
    all_combinations = [list(i.flatten()) for i in all_combinations]
    all_combinations.remove([0]*N)
    print("Columns; Precision; Recall; F1; Accuracy;Product")
    for combin in all_combinations:
        # combin = [0, 0, 0, 0, 1, 0, 0, 0]
        # print("-------------------------",,"----------------------------------")
        # print(np.array(combin)==1)
        # print(np.array(df_tmp.columns[:-1])[np.array(np.array(combin)==1)])
        
        # print("COMBIN", combin)
        train_X_abelation, train_Y_abelation, params = feature_scaling(train_X, train_Y,[],5,combin)
        test_X_abelation, test_Y_abelation, params = feature_scaling(test_X, test_Y,params,5,combin)
        # print(train_X.shape, train_X_abelation.shape)
        # print(train_X.shape, test_X.shape)
        

        outlier_matrix = get_outlier_matrix(train_X_abelation, train_Y_abelation)
        categories = [0,1,0,0,1,1,1,0]
        models = awesome_mixture_nb(train_X_abelation, train_Y_abelation, test_X_abelation, test_Y_abelation, categories, outlier_matrix)
        Precision, Recall, F1, Accuracy = calc_score(models, test_X_abelation, test_Y_abelation,categories)
        print(list(np.array(df_tmp.columns[:-1])[np.array(combin)==1]),Precision, Recall, list(F1), Accuracy, Accuracy*list(F1)[0]*list(F1)[1],sep=";")
        
        # real_test_x, real_test_y = read_raw_test("diabetes_testing.csv",means)
        # print(len(models), real_test_x.shape, real_test_y.shape, len(categories))
        # print()
        # print("ON UNSEEN DATA")
        # calc_score(models, real_test_x, real_test_y, categories)

    
    # train_categoNB(train_X, train_Y, test_X, test_Y)
    # train_multi_nb(train_X, train_Y, test_X, test_Y)
    # train_ComplementNB(train_X, train_Y, test_X, test_Y)
    # plot_PCA_data(train_X, train_Y)
    # plot_PCA_data(test_X, test_Y)
    
    # PCA_COMP = 3
    # train_X, train_Y, test_X, test_Y = PCA_train(train_X, train_Y, test_X, test_Y, PCA_COMP)
    # print("AFTER PCA:")
    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    # train_data(train_X, train_Y, test_X, test_Y)
    
    # see_LDA(train_X, train_Y)
    # see_LDA(test_X, test_Y)




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