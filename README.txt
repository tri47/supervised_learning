Steps to run the code:
1. Set up the environment with the libraries in enviroment.yml file (using conda or pip)
The libraries include:
- python=3.7.2
- matplotlib=3.0.3
- numpy=1.16.2
- pandas=0.24.1
- mlrose==1.3.0
2. Run the program to produce the graphs for 2 problemes:
For bank marketing problem:
    python supervised_learning.py  --problem="bank" --LearningCurve=1 --DT=1 --Ada=1 --Bagging=1 --KNN=1 --SVM=1 --NN=1
for diseased tree problem
    python supervised_learning.py  --problem="tree" --LearningCurve=1 --DT=1 --Ada=1 --Bagging=1 --KNN=1 --SVM=1 --NN=1
Change the variable such as DT, Ada from 1 to 0 to show only the results required.

The data sets are already included. They were obtained from:

1. http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
2. http://archive.ics.uci.edu/ml/datasets/wilt

Reference:
1.	Moro, S, Cortez, P. and Rita, P.. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014. 
2.	Johnson, B., Tateishi, R., Hoan, N., 2013. A hybrid pansharpening approach and multiscale object-based image analysis for mapping diseased pine and oak trees. International Journal of Remote Sensing, 34 (20), 6969-6982.
