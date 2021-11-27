# -*- coding: utf-8 -*-
"""PerceptronClassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lLTEmsumAW5Eo-rH1RMR67KYBkqjOsor
"""

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment=None
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

df=pd.read_csv("diabetes_training.csv")
df.head()

#Preprocessing
#Changing 0 values in Glucose, BloodPressure and BMI
df_nodiab=df[df["Outcome"]==0]
df_diab=df[df["Outcome"]==1]
glucose_count_0=0
glucose_count_1=0
bloodpressure_count_0=0
bloodpressure_count_1=0
bmi_count_0=0
bmi_count_1=0
glucose=df_nodiab["Glucose"]
bloodpressure=df_nodiab["BloodPressure"]
bmi=df_nodiab["BMI"]
for i in glucose:
  if i==0:
    glucose_count_0=glucose_count_0+1
for i in bloodpressure:
  if i==0:
    bloodpressure_count_0=bloodpressure_count_0+1
for i in bmi:
  if i==0:
    bmi_count_0=bmi_count_0+1
avg_glucose_0=np.mean(glucose)*(glucose.shape[0]/(glucose.shape[0]-glucose_count_0))
avg_bloodpressure_0=np.mean(bloodpressure)*(bloodpressure.shape[0]/(bloodpressure.shape[0]-bloodpressure_count_0))
avg_bmi_0=np.mean(bmi)*(bmi.shape[0]/(bmi.shape[0]-bmi_count_0))
glucose=df_diab["Glucose"]
bloodpressure=df_diab["BloodPressure"]
bmi=df_diab["BMI"]
for i in glucose:
  if i==0:
    glucose_count_1=glucose_count_1+1
for i in bloodpressure:
  if i==0:
    bloodpressure_count_1=bloodpressure_count_1+1
for i in bmi:
  if i==0:
    bmi_count_1=bmi_count_1+1
avg_glucose_1=np.mean(glucose)*(glucose.shape[0]/(glucose.shape[0]-glucose_count_1))
avg_bloodpressure_1=np.mean(bloodpressure)*(bloodpressure.shape[0]/(bloodpressure.shape[0]-bloodpressure_count_1))
avg_bmi_1=np.mean(bmi)*(bmi.shape[0]/(bmi.shape[0]-bmi_count_1))
arr=df.values
for i in arr:
  if i[8]==0:
    if i[1]==0:
      i[1]=avg_glucose_0
    if i[2]==0:
      i[2]=avg_bloodpressure_0
    if i[5]==0:
      i[5]=avg_bmi_0
  else:
    if i[1]==0:
      i[1]=avg_glucose_1
    if i[2]==0:
      i[2]=avg_bloodpressure_1
    if i[5]==0:
      i[5]=avg_bmi_1
df=pd.DataFrame(arr,index=df.index,columns=df.columns)
df.head()

x_columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
df_x=df[x_columns]
scaler=StandardScaler()
scaled=scaler.fit_transform(df_x)
df_x=pd.DataFrame(scaled,index=df_x.index,columns=df_x.columns)
df[x_columns]=df_x
df.head()

X=df[x_columns].values
Y=df["Outcome"].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

model=Perceptron(class_weight="balanced")
model.fit(X_train,Y_train)
Y_hat=model.predict(X_test)
print("Accuracy: ",accuracy_score(Y_test,Y_hat))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, Y_hat)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, Y_hat)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, Y_hat)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, Y_hat)
print('F1 score: %f' % f1)

confusion_matrix(Y_test, Y_hat)

df_train=pd.read_csv("diabetes_training.csv")
df_train.head()

X_train=df_train[x_columns].values
Y_train=df_train["Outcome"].values

model=Perceptron()
model.fit(X,Y)

df_test=pd.read_csv("diabetes_testing.csv")
df_test.head()

X_test=df_test[x_columns].values
Y_test=df_test["Outcome"].values

Y_hat=model.predict(X_test)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, Y_hat)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, Y_hat)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, Y_hat)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, Y_hat)
print('F1 score: %f' % f1)

Y_hat

"""Hyper-parameter Tuning"""

from sklearn.model_selection import GridSearchCV

param_grid = {'l1_ratio': [0.1,0.15,0.3,0.5,0.7,0.9],

              'alpha':[1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001,
                                   1e-05, 1e-06, 1e-07, 1e-07, 1e-08, 1e-09,
                                   1e-10, 1e-11],
              'tol':[1e-3,1e-2,1e-6],
              'max_iter':[1000,10000,100000,10000000],
                'early_stopping':[True,False],
              'penalty': [None,'l2','l1','elasticnet']}

grid = GridSearchCV(Perceptron(class_weight="balanced"), param_grid, refit = True, verbose = 3,n_jobs=10)

# grid.fit(X_train, Y_train)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(grid, X_train, Y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

print(grid.best_estimator_)

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(Y_train, grid.predict(X_train))

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_train,  grid.predict(X_train)))

import seaborn as sns
sns.heatmap(cf_matrix, annot=True, fmt='g')

"""# RandomUnderSampler"""

from imblearn.under_sampling import RandomUnderSampler # Up-sample or Down-sample

rus = RandomUnderSampler(random_state=42)
X_res, Y_res = rus.fit_resample(X, Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.005)

print("Training data set shape : ", X_train.shape, Y_train.shape)
print("Test data set shape : ", X_test.shape, Y_test.shape)

model=Perceptron()
model.fit(X_train,Y_train)
Y_hat=model.predict(X_test)
print("Accuracy: ",accuracy_score(Y_test,Y_hat))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, Y_hat)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, Y_hat)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, Y_hat)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, Y_hat)
print('F1 score: %f' % f1)

confusion_matrix(Y_test, Y_hat)
