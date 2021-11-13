# FML_Final_Project
Course Project of FML  
Heyy :)))  
Google Colab link: https://colab.research.google.com/drive/1YppdFQ5tSic-qJUKJxPRhN5O2WLyIFbS?usp=sharing  
  
LOG:

*Sep 29*: Trying Linear SVC(Support Vector Classifier) with l1 on unprocessed data: Gets 80% accuracy :)
*Sep 30*: Analyzing Data  
**Glucose Column**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test  
![image](https://user-images.githubusercontent.com/88259695/135402142-d537f6c7-5567-4cff-b139-7b9553f2929c.png)   
Observation: In the data we have all the glucose level is <200mg/dL. 44-199 to be precise  
As we move closer to 199, more people are diabetic  
In the prediabetic category, 193 samples are present. Out of which 133 are diabetic i.e. around 68.9%  

Check this sample: 9,	140,	94,	0,	0,	32.7,	0.734,	45,	1
Patient is diabeteic
9 preg
140 oral Glucose [Glucode Tolerance Test(GTT)]
94 Blood pressure
no info on skin thickness and insulin
BMI 32.7 hence Obese
0.734 Diabetes Pedigree Function
45 age

**Diabetes Pedigree Function  (DPF)**
For formula of the function check references, heres what you can conclude looking at DPF  
![image](https://user-images.githubusercontent.com/88259695/135406674-f6406b53-f40d-4a47-80ba-aae90650066c.png)
  
  
**BMI**  
![image](https://user-images.githubusercontent.com/88259695/135403626-09b78e96-fcc2-44ac-9fab-4fae502d3bd4.png)

REFERENCES:  
*About Diabetes Pedigree Function on page 2, also talks about why and how the datacollection was done*: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf

DEFINITIONS:  
**Diabetes mellitus (DM)**, commonly known as _just diabetes_, is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period of time. Symptoms often include frequent urination, increased thirst and increased appetite. If left untreated, diabetes can cause many health complications.


*Oct 1*: Revised mathematical proof of SVM  
*Oct 2*: Insulin 2-Hour serum insulin (mu U/ml)  
Tried PCA, reduced accuracy to 77.9%  
Tried filling 0s, reduced accuracy    
*Oct 4*: Since very less test data, accuracies we obtain won't be a correct measure  
If I change the test set I go from 80% to 70%. It could also happen that a technique may give lower accuracy on one test set but does it mean it didnt perform well?   
Soln: Cross Validation
It turns out that there are some techniques to handle imbalance
Those are: upsampling,downsampling(no way we can afford it), synthesis technique(SNORT) and class balancing

Class balancing used :)

Accuracy is not a correct measure, Instead use precision recall
Problem of higher false positives

Tuning the hyper parameters c and gamma
One of the good ones: check in notebook
SNORT can be used for upsampling

Preprocessing isn't straight forward because of the time duration between initial tests and final results
Grid Search incorporates cross validation (5 fold)
![image](https://user-images.githubusercontent.com/88259695/136698194-021a97d4-7ccd-4761-85b5-2cba041be00d.png)


Imputation can be used for handling missing values:
https://scikit-learn.org/stable/modules/impute.html


What are the most important features?-Feature Selection:
https://scikit-learn.org/stable/modules/feature_selection.html


Following Preprocessing is done:
1) Univariate Imputation (Tried MultiVariate made th)
2) Outliers Removed
3) Scaling


Correlation of features:
![image](https://user-images.githubusercontent.com/88259695/141613718-1a342507-3794-4b83-9f36-3afec1929e22.png)


Results(with all features):

**Report On 10% of data:**
Hyper-parameters:
clf=svm.SVC(C=1000, break_ties=False, cache_size=100000, class_weight='balanced',
    coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.001,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
    
![image](https://user-images.githubusercontent.com/88259695/141613775-915d8744-9ffd-4b9f-8d11-ecab38bf0041.png)

**Report On 30% of data**
Hyper-parameters:
SVC(C=1000, break_ties=False, cache_size=100000, class_weight='balanced',
    coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.0001,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)

![image](https://user-images.githubusercontent.com/88259695/141614028-a891c282-ea12-4ad5-825e-4084d9d9376d.png)


Tried PCA-Did not work for feature reduction in SVM

Two least important features (after checking the weights and re running SVM):
Blood Pressure and Skin Thickness

Applying Genetic Algo for feature selection for 400 generations shows following as importnat features:
Glucose, SkinThickness and 	DiabetesPedigreeFunction
Report of Genetic Algorithm:
![image](https://user-images.githubusercontent.com/88259695/141613908-02961999-8e62-4c62-8a4e-7487a11336ee.png)
![image](https://user-images.githubusercontent.com/88259695/141613913-a42f11e3-8fa1-4875-8b06-96100533e0c0.png)


**IGNORE THE PART BELOW, NEED TO DO A BETTER ABLATION STUDY**

Both the above analysis focuses on accuracy. But here Precision of 1 is really important(that is number of false positives)
With an ablation study, the most important featues that contibute to 1-precision are: Blood Pressure and DiabetesPedegreeFunction
and most imp features that contributes to accuracy are: Glucose,DiabetesPedigreeFunction,Age

With only Blood Pressure and DiabetesPedegreeFunction on 30% of data
