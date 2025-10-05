# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 20:07:05 2025

@author: Ranjan Segu
"""

                       ####Scikit_Learn_Functions####
                       
            
#Loading_Data

import numpy as np

x = np.random.random((10, 5))
y = np.array(["M", "M", "F", "F", "M", "F", "M", "M", "F", "F"])
x[x < 0.7] = 0


#Train_Test_Data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


#Preprocessing_Data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler () .fit(x_train)
standardized_x_train = scaler.transform (x_train)
standardized_x_test = scaler.transform (x_test)

print("Before scaling:")
print("Train mean:", np.mean(x_train, axis=0))
print("Train std: ", np.std(x_train, axis=0))
print("Test mean: ", np.mean(x_test, axis=0))
print("Test std:  ", np.std(x_test, axis=0))

print("\nAfter scaling:")
print("Train mean:", np.mean(standardized_x, axis=0))
print("Train std: ", np.std(standardized_x, axis=0))
print("Test mean: ", np.mean(standardized_x_test, axis=0))
print("Test std:  ", np.std(standardized_x_test, axis=0))


from sklearn.preprocessing import Normalizer
scaler = Normalizer () .fit(x_train)
normalized_x_train = scaler.transform (x_train)
normalized_x_test = scaler.transform (x_test)

from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0) .fit(x)
binary_x = binarizer.transform (x)

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = 0, strategy="mean")
imp.fit_transform(x)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(x)

#Algorithms#

#Linear_Regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x, y)
y_pred = lr.predict(x_test)


#Support_Vector_Machines

from sklearn.svm import SVC
svc = SVC(kernel = "linear")

svc.fit(x_train, y_train)
y_pred = svc.predict(np.random.random((2, 5)))


#Naive_Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#KNN

from sklearn import neighbors
model = neighbors.KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train, y_train)
y_pred = knn.predict_proba(x_test)



#Principal_Component_Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

pca_model = pca.fit_transform(x_train)
y_pred = kmeans.predict(x_test)


#KMeans

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x_train)


#Classification_Metrics

knn.score(x_test, y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
y_train_enc = enc.fit_transform(y_train)
y_test_enc = enc.transform(y_test)
model.fit(x_train, y_train)
y_pred_enc = model.predict(x_test)
print(classification_report(y_test, y_pred_enc))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_enc))


#Regression_Metrics

from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2]
mean_absolute_error(y_true, y_pred)

from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2]
mean_squared_error(y_true, y_pred)

from sklearn.metrics import r2_score
r2_score(y_true, y_pred)


#Clustering_Metrics

from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_true, y_pred)

from sklearn.metrics import homogeneity_score
homogeneity_score(y_true, y_pred)

from sklearn.metrics import v_measure_score
v_measure_score(y_true, y_pred)


#Cross_Validation

from sklearn.model_selection import cross_val_score
print(cross_val_score(knn, x_train, y_train, cv=4))
print(cross_val_score(lr, x, y_pred, cv=2))

#Tune_Model

from sklearn.model_selection import GridSearchCV
params = {"n_neighbors": np.arange(1,3), "metric":["euclidean", "cityblock"]}
grid = GridSearchCV(estimator = knn, param_grid=params, cv = 3)
grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)


#Randomized_Parameter_Optimization

from sklearn.model_selection import RandomizedSearchCV
params = {"n_neighbors": np.arange(1,5), "weights":["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn, param_distributions=params, cv = 4, n_iter=8, random_state=5)
rsearch.fit(x_train, y_train)
print(rsearch.best_score_)
