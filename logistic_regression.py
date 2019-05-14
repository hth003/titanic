# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y_train = dataset_train.iloc[:, 1].values
dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy  = "mean")
imputer = imputer.fit(X_train[:,2:4])
X_train[:,2:4] = imputer.transform(X_train[:,2:4])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
