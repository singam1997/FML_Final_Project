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
