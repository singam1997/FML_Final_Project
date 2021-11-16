# FML_Final_Project
Course Project of FML

PCA effect of accuracy on NB:
No of components---avg accuracy over 50 times random split
1---0.689531
2---0.752344
3---0.762344
4---0.75125
5---0.759375
6---0.76625

PCA dimensionality reduction to 3 dimensions. We get the image PCA_dim_red.png. One can run the function plot_PCA_data("input csv file") to generate the graph.

AdaBoost: Observe that as the number of estimators increase the acc increases initially and then suddenly decreases!

0 values in-
Pregnancies: 111
Glucose: 5
BloodPressure: 35
SkinThickness: 227
Insulin: 347
BMI: 11
DiabetesPedigreeFunction: 0
Age: 0

0 values in Glucose, BloodPressure, SkinThickness, Insulin and BMI replaced with mean values in the read_raw_data function.
The best accuracy reported with GNB on this is 81.45%.
After performing PCA, the accuracy remains almost the same (on all possible PCA dimensions 1,2,3,4,5,6,7 and 8.
Maximum accuracy reported on adaboost is also 80%

Feature Scaling: Unexpectedly the accuracy seems to drop after mean normalization, standardization and shifting of origin, on GNB to 68%-75%. But accuracy on AdaBoost remains similar to without feature scaling.

Two plots here: The one with the feature scaling has highest accuracy 76% when there is only one weak learner :(. The one without feature scaling has highest accuracy of 77% when there is only one weak learner i.e. the GNB :( in the adaboost. The plot also seems to oscillate after some threshold of weak learners :?
