# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:09:15 2022

@author: prit patel exercise 2 
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1.	Load the data into a pandas 
data_prit_df2 = pd.read_csv(r"C:\Users\16478\Documents\Prit Comp 171\Supervised Learning\SVM Lab\breast_cancer.csv")

#3.
data_prit_df2['bare'].replace('?', np.nan, inplace=True)
data_prit_df2['bare'] = data_prit_df2['bare'].astype('float')

data_prit_df2.drop('ID', axis=1, inplace=True)

X = data_prit_df2.drop('class', axis=1)
y = data_prit_df2['class']
X, y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

#6.
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
scaler = StandardScaler()
scaler.fit(X_train)

#a. missing values with the median of the column
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(X_train)
#b.Scale the data  u
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#7.
from sklearn.pipeline import Pipeline
num_pipe_prit = Pipeline([('imputer', imp), ('scaler', scaler)])

#8.Created a new Pipeline
from sklearn.svm import SVC
pipe_svm_prit = Pipeline([('num_pipe', num_pipe_prit), ('svc', SVC(random_state=48))])
pipe_svm_prit

#10. grid search parameters 
param_grid_svm = [{
    'svc__C': [0.01,0.1, 1, 10, 100], 
    'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'svc__kernel': ['linear', 'rbf','poly'],
    'svc__degree':[2,3]
    }]

param_grid_svm

#12. grid search object 
from sklearn.model_selection import GridSearchCV
grid_search_prit = GridSearchCV(estimator=pipe_svm_prit, param_grid=param_grid_svm, refit=True, verbose=3, cv=5, scoring='accuracy')

#14. Fiting of training data to the gird search object
grid_search_prit.fit(X_train_scaled, y_train)

#15. best parameters 
print(grid_search_prit.best_params_)
print(grid_search_prit.best_score_)

#16. Printout the best  
print(grid_search_prit.best_estimator_)

#17.Fit the test data the grid search object
grid_search_prit.fit(X_test_scaled, y_test)

#18. accuracy score
print(grid_search_prit.score(X_test_scaled, y_test))

#19.
best_model_prit = grid_search_prit.best_estimator_

#20.Save model using the joblib  
import joblib
joblib.dump(best_model_prit, 'best_model_prit.pkl')

#21.Save the full pipeline 
joblib.dump(pipe_svm_prit, 'pipe_svm_prit.pkl')

