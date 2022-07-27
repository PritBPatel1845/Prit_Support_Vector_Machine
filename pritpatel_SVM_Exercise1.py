# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:12:11 2022

@author: prit patel exercise 1
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#1.	Load the data 
data_prit = pd.read_csv(r"C:\Users\16478\Documents\Prit Comp 171\Supervised Learning\SVM Lab\breast_cancer.csv")
data_prit
data_prit.head()
data_prit.columns
data_prit.dtypes
data_prit.info()
data_prit.describe()
data_prit.min()
data_prit.max()
data_prit.mean()
data_prit.median()


#3.
data_prit['bare'].replace('?', np.nan, inplace=True)
data_prit['bare'] = data_prit['bare'].astype('float')
data_prit['bare'].isnull().sum()


#4.	Fill any missing data 
data_prit['bare'].fillna(data_prit['bare'].median(), inplace=True)
data_prit['bare'].isnull().sum()


#5.	Drop the ID column
data_prit.drop('ID', axis=1, inplace=True)


#6
data_prit.hist(figsize=(10,10))
plt.show()


#7.	Separate the features
X = data_prit.drop('class', axis=1)
y = data_prit['class']
X
y

#8.	Spliting data into train 80% train and 20% test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)


# ## Build Classification Models


#9.
from sklearn.svm import SVC
clf_linear_prit = SVC(kernel='linear', C=0.1)
clf_linear_prit.fit(X_train, y_train)




#10 Print out two accuracy score  
print(int(clf_linear_prit.score(X_train, y_train) * 100) ,'%')
print(int(clf_linear_prit.score(X_test, y_test) * 100) ,'%')



#11. Generate the accuracy matrix 
from sklearn.metrics import confusion_matrix
cm_linear_prit = confusion_matrix(y_test, clf_linear_prit.predict(X_test))
print(cm_linear_prit)




from sklearn.svm import SVC
clf_linear_prit = SVC(kernel='rbf')
clf_linear_prit.fit(X_train, y_train)

print(int(clf_linear_prit.score(X_train, y_train) * 100) ,'%')
print(int(clf_linear_prit.score(X_test, y_test) * 100) ,'%')

from sklearn.metrics import confusion_matrix
cm_linear_prit = confusion_matrix(y_test, clf_linear_prit.predict(X_test))
print(cm_linear_prit)


from sklearn.svm import SVC
clf_linear_prit = SVC(kernel='sigmoid')
clf_linear_prit.fit(X_train, y_train)

print(int(clf_linear_prit.score(X_train, y_train) * 100) ,'%')
print(int(clf_linear_prit.score(X_test, y_test) * 100) ,'%')

from sklearn.metrics import confusion_matrix
cm_linear_prit = confusion_matrix(y_test, clf_linear_prit.predict(X_test))
print(cm_linear_prit)



from sklearn.svm import SVC
clf_linear_prit = SVC(kernel='poly')
clf_linear_prit.fit(X_train, y_train)

print(int(clf_linear_prit.score(X_train, y_train) * 100) ,'%')
print(int(clf_linear_prit.score(X_test, y_test) * 100) ,'%')

from sklearn.metrics import confusion_matrix
cm_linear_prit = confusion_matrix(y_test, clf_linear_prit.predict(X_test))
print(cm_linear_prit)

