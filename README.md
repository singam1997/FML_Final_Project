# FML_Final_Project
Course Project of FML
Data has 500 0s and 268 1s
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

Gaussian NB seems to perform the best followed by multinomialNB (on data after removing BMI and Diabetes) followed closely by ComplementNB and Categorical NB

Unique values after doing np.round(data):
preganacies: 17
glucose: 135
blood pressure: 47
skin thickness: 50
insulin: 186
BMI: 39 : without np.round 247
DiabetesPedigreeFunction: 3 : without np.round 517
Age: 52

Reported Values:
Mix_acc: 0.7792207792207793
PRECISION: 0.6984126984126984
RECALL: 0.7457627118644068

Awesome_Mix_acc: 0.7987012987012987
Awesome_PRECISION: 0.7258064516129032
Awesome_RECALL: 0.7627118644067796

Increasing weight of 1 samples increases recall
Increasing weight of 0 samples increases precision

sample weights just effect the means and std deviations (I think from the source code)
new_mu = np.average(X, axis=0, weights=sample_weight)
new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
I guess here is where the sample weights in the source code are used, it becomes weighted mean and weighted variance.

BTW I have uploaded test data here! USE THIS TO GET THE ACCURACY, RECALL, PRECISION! OR GIVE ANOTHER TEST DATA FILE!!! FOR THE SAME!

best scores recorded on naive bayes:
on dev set

---------------------------

Awesome_Mix_acc: 0.8091603053435115
Awesome_PRECISION: 0.7555555555555555
Awesome_RECALL: 0.7083333333333334

--------------------------

ON UNSEEN TEST DATA

---------------------------

Awesome_Mix_acc: 0.8362068965517241
Awesome_PRECISION: 0.75
Awesome_RECALL: 0.7297297297297297

--------------------------

