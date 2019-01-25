# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 22:09:48 2019

@author: Anant Vignesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#scikit learn library 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestRegressor
#scikit learn library


#-----------------CUSTOM FUNSTIONS------------------------#

def plot_corr(df):
    corr=df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

#-----------------CUSTOM FUNSTIONS------------------------#
    
    
#-----------------READING DATASET------------------------#

dataset = pd.read_csv('E:/MS COMPUTER SCIENCE/MS PROJECTS/OWN PROJECTS/Titanic/Data/train.csv')
df = dataset
df.info()

#-----------------READING DATASET------------------------#


#-----------------DATA PREPROCESSING---------------------#

#REMOVING NAN VALUES FROM THE DATASET
from sklearn.impute import SimpleImputer as Im
imputer = Im(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(df.iloc[:, 5:6])
df.iloc[:, 5:6] = imputer.transform(df.iloc[:, 5:6])


#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder
labelencoder_df = LabelEncoder()
df['Sex'] = labelencoder_df.fit_transform(df['Sex']) #Converting Sex To Numerical
df['Pclass'] = labelencoder_df.fit_transform(df['Pclass'])
df['Embarked'] = labelencoder_df.fit_transform(df['Embarked'].astype(str))


#PLOT CORRELATION MATRIX AND GRAPH TO FIND DEPENDENT VARIABLES
cor_mat = df.corr()
plot_corr(df)


#ONEHOT ENCODING OF CATEGORICAL DATA
#df = pd.get_dummies(df, columns=["Sex"], prefix=["Sex"])
#df = pd.get_dummies(df, columns=["Pclass"], prefix=["Pclass"])
#df = pd.get_dummies(df, columns=["Embarked"], prefix=["Embarked"])


#DROPPING UNWANTED COLUMNS
#df.drop(['Cabin','Ticket','Name'], axis=1, inplace=True)
df.drop(['Cabin','Ticket','Name'], axis=1, inplace=True)

#-----------------DATA PREPROCESSING---------------------#


#-----------SPLITTING TRAINING AND TESTING DATA----------#

#SEPERATE LABEL COLUMN FROM FEATURE COLUMNS
df_label = df['Survived'].values
df.drop(['Survived'], axis=1, inplace=True)
df_feature = df.values


#SPLIT TRAINING SET AND TESTING SET
from sklearn.model_selection._split import train_test_split
feature_train,feature_test,label_train,label_test = train_test_split(df_feature, df_label, test_size=0.20)

#-----------SPLITTING TRAINING AND TESTING DATA----------#

#PLOT CORRELATION MATRIX AND GRAPH TO FIND DEPENDENT VARIABLES
cor_mat = df.corr()
plot_corr(df)

#----------------DATA MODELLING---------------------------#

#NAIVE BAYESIAN
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
clf = GaussianNB()
clf1= MultinomialNB()
clf2=BernoulliNB()
clf.fit(feature_train,label_train)
clf1.fit(feature_train,label_train)
clf2.fit(feature_train,label_train)
accuracy = dict()
predicted_values_GB=clf.predict(feature_test)
predicted_values_NB=clf1.predict(feature_test)
predicted_values_BB=clf2.predict(feature_test)


#GRADIENT BOOSTING ALGORITHM
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model.fit(feature_train, label_train)
predicted_values_GBA = model.predict(feature_test)


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(feature_train,label_train)
model_LR.score(feature_train,label_train)
#Equation coefficient and Intercept
print('Coefficient: \n', model_LR.coef_)
print('Intercept: \n', model_LR.intercept_)
predicted_values_LR = model_LR.predict(feature_test)

#SVM (SUPPORT VECTOR MACHINE)
from sklearn import svm 
model_svm = svm.SVC()
model_svm.fit(feature_train,label_train)
model_svm.score(feature_train,label_train)
predicted_values_svm = model_svm.predict(feature_test)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier()
model_RF.fit(feature_train,label_train)
predicted_values_RF = model_RF.predict(feature_test)

#LINEAR REGRESSION
#from sklearn.linear_model import LinearRegression
#clf = LinearRegression()
#clf.fit(feature_train,label_train)
#predicted_values_LG = clf.predict(feature_test)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(feature_train,label_train)
predicted_values_DT = clf.predict(feature_test)

#K-NN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(feature_train,label_train)
predicted_values_KNN = clf.predict(feature_test)

#----------------DATA MODELLING---------------------------#

#----------------ACCURACY CHECK/CALCULATION---------------------------#

accuracy['Gaussian'] = accuracy_score(predicted_values_GB,label_test)*100
accuracy['MultinomialNB'] = accuracy_score(predicted_values_NB,label_test)*100
accuracy['BernoulliNB'] = accuracy_score(predicted_values_BB,label_test)*100
accuracy['GBA'] = accuracy_score(predicted_values_GBA,label_test)*100
accuracy['LogisticReression'] = accuracy_score(predicted_values_LR,label_test)*100
accuracy['SVM'] = accuracy_score(predicted_values_svm,label_test)*100
accuracy['RandomForest'] = accuracy_score(predicted_values_RF,label_test)*100
#accuracy['LinearRegression'] = accuracy_score(predicted_values_LG,label_test)*100
accuracy['DecisionTree'] = accuracy_score(predicted_values_DT,label_test)*100
accuracy['KNN'] = accuracy_score(predicted_values_KNN,label_test)*100
accuracy['Max_accuracy'] = 100
accuracy=pd.DataFrame(list(accuracy.items()),columns=['Algorithm','Accuracy'])
print(accuracy)
sns.lineplot(x='Algorithm',y='Accuracy',data=accuracy)

#----------------ACCURACY CHECK/CALCULATION---------------------------#

#----------------WRITE PREPROCESSED DATA AS A CSV FILE----------------#

df = pd.DataFrame(X)
df.to_csv('DataTitanic.csv', header = None, index = None)

#----------------WRITE PREPROCESSED DATA AS A CSV FILE----------------#
