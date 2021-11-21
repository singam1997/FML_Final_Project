import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(X, Y):
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    df = pd.concat([df_x,df_y],axis=1)

    train, test = train_test_split(df, test_size=0.15)#,random_state=random_n)
    
    train_Y = np.array(train)[:,-1]
    train_X = np.array(train)[:,:-1]

    test_Y = np.array(test)[:,-1]
    test_X = np.array(test)[:,:-1]

    return train_X, train_Y, test_X, test_Y


def read_raw_data(file_name):
    df = pd.read_csv(file_name)
    Y = df["Outcome"].to_frame()
    X = df.drop("Outcome",axis=1)

    return X.values, Y.values.flatten()


if __name__=="__main__":
    X,Y = read_raw_data("diabetes.csv")
    train_X, train_Y, test_X, test_Y = split_data(X,Y)
    
    df = pd.read_csv("diabetes.csv")
    
    training_data = np.hstack((train_X,train_Y.reshape(-1,1)))
    df_training = pd.DataFrame(training_data)
    df_training.columns = df.columns
    df_training.to_csv('diabetes_training.csv',index=False)

    testing_data = np.hstack((test_X,test_Y.reshape(-1,1)))
    df_testing = pd.DataFrame(testing_data)
    df_testing.columns = df.columns
    df_testing.to_csv('diabetes_testing.csv',index=False)