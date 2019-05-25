# Titanic

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y_train = dataset_train.iloc[:, 1].values
dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer_age = SimpleImputer(missing_values=np.nan, strategy  = "mean")
imputer_age = imputer_age.fit(X_train[:,2:4])

X_train[:,2:4] = imputer_age.transform(X_train[:,2:4])
X_test[:,2:4] = imputer_age.transform(X_test[:,2:4])

imputer_em = SimpleImputer(missing_values=np.nan, strategy  = "most_frequent")
imputer_em = imputer_em.fit(X_train[:,5:7])
X_train[:,5:7] = imputer_em.transform(X_train[:,5:7])
X_test[:,5:7] = imputer_em.transform(X_test[:,5:7])

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:,1] = labelencoder_X.fit_transform(X_train[:,1]) #encode gender
X_test[:,1] = labelencoder_X.fit_transform(X_test[:,1]) #encode gender
ct = ColumnTransformer(
        [('one_hot', OneHotEncoder(sparse=False), [6]),], remainder='passthrough') 
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_traint, X_cval, y_traint, y_cval = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

# Fitting Kernel SVM to the Training set
'''from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_traint, y_traint)'''
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_cval, y_pred)

output=pd.DataFrame({'PassengerId':dataset_test.iloc[:, 0].values,'Survived':y_pred})
output.to_csv('submission.csv', index=False)

