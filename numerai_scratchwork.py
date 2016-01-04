## numerai scratch work

import pandas as pd
import numpy as np
import os
from __future__ import division
from sklearn.cross_validation import train_test_split

#work machine
os.chdir("Dropbox/Projects/Numerai/")

df = pd.read_csv("numerai_datasets/numerai_training_data.csv")

#train/test split
np.random.seed(1234)
data = train_test_split(df)
X_train = data[0].iloc[:, :15]
y_train = data[0].iloc[:, 16]
X_test = data[1].iloc[:, :15]
y_test = data[1].iloc[:, 16]

# create additional features
# can try grid multiply / powers; addition/subtraction don't make much sense

# try some models
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 20)
rf.fit(X_train.iloc[:, :14], y_train)
rf_score = rf.score(X_test.iloc[:, :14], y_test)
#score ~0.505

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train.iloc[:, :14], y_train)
lr_score = lr.score(X_test.iloc[:, :14], y_test)
#score ~0.526
