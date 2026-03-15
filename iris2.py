import pandas as pd
import numpy as np
iris = pd.read_csv("/content/iris.csv")
iris

iris.isnull().sum()

iris = iris.dropna()
iris.isnull().sum()

iris.dtypes

data = iris.drop(["species"], axis=1)
data

label=iris["species"]
label

data = data.to_numpy()
data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.3, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import pickle
model=pickle.dump(model, open("model_svm.pkl", "wb"))

import pickle
model=pickle.load(open("model_svm.pkl", "rb"))
model

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

model.predict([[1.0, 2.3, 1.2, 1]])

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import sklearn
print(sklearn.__version__)

