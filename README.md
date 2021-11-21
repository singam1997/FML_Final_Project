# Decision Tree & Random Forest

Decision Tree :
   1. https://dhirajkumarblog.medium.com/top-5-advantages-and-disadvantages-of-decision-tree-algorithm-428ebd199d9a

Random Forest
  1. https://www.mygreatlearning.com/blog/random-forest-algorithm/
  2. http://theprofessionalspoint.blogspot.com/2019/02/advantages-and-disadvantages-of-random.html

Breaking the curse of small datasets in Machine Learning
https://towardsdatascience.com/breaking-the-curse-of-small-datasets-in-machine-learning-part-1-36f28b0c044d

## Summary

Training set size : 80%(614 instances) \
Test set size : 20%(154 instances) 

| Sr. | Preprocessing | Best Decision Tree Model Parameter | Decision Tree Accuracy (%) | Best Random Forest Model Parameter | Random Forest Accuracy (%) | Jupyter notebook | PDF |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| 0 |  No | criterion="entropy", splitter="best", max_depth=3 | 79.87 | n_estimators = 1000, criterion='entropy' | 82.47  | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V0.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V0.pdf) |
| 1 | Replace BMI, BP, ST with mean | criterion="entropy", splitter="best", max_depth=3 | 79.22 | n_estimators = 100, random_state = 42, max_depth = 8 | 82.47 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V1.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V1.pdf) |
| 2 | Replace BMI, BP, ST with median | criterion="entropy", splitter="best", max_depth=3 | 79.22 | n_estimators = 1000, criterion='entropy' | 80.52 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V2.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V2.pdf) |
| 3 | Replace All zero features with mean |criterion="entropy", splitter="best", max_depth=3 | 79.22 | n_estimators = 100, random_state = 42, max_depth = 8, criterion='entropy' | 81.82 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V3.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V3.pdf) |
| 4 | Replace All zero features with median | criterion="entropy", splitter="best", max_depth=3 | 79.22 | n_estimators = 1000, criterion='entropy' | 79.87 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V4.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V4.pdf) |
| 5 | Replace All zero features with mean & compute_class_weight | criterion="entropy", splitter="random", max_depth=3, class_weight='balanced' | 75.97 | n_estimators = 1000, random_state = 42, max_depth = 8, class_weight='balanced' | 82.47 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V5.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V5.pdf) |
| 6 | Replace all zero features with mean & RandomUnderSampler | criterion="entropy", splitter="random", max_depth=8 | 75.00 | n_estimators = 1000, random_state = 42, max_depth = 8, criterion='entropy' | 84.26 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V6.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V6.pdf) |
| 7 | Replace all zero features with mean & RandomOverSampler | criterion="entropy", splitter="random" | 82.00 | n_estimators = 100 | 83.00 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V7.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V7.pdf) |
| 8 | Replace All zero features with mean & compute_class_weight & RandomOverSampler | criterion="gini", splitter="best", max_depth=8, class_weight='balanced' | 75.32 | criterion='entropy', n_estimators = 100, random_state = 42, max_depth = 8, class_weight='balanced' | 82.47 | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V8.ipynb) | [link](https://github.com/singam1997/FML_Final_Project/blob/kamal/Decision%20Tree%20%26%20Random%20Forest%20V8.pdf) |
