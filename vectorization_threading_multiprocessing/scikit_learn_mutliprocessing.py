"""
Created on Jan 17, 2022

@author: inot
"""

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

X, y = datasets.make_classification(
    n_samples=10000, n_features=50, n_informative=20, n_classes=10)

# SERIAL EXECUTION
start = time.time()
model = RandomForestClassifier(n_estimators=500)
model.fit(X, y)
print('Time:', time.time() - start)

# PARALLEL EXECUTION BY USING MULTIPROCESSING
start = time.time()
model = RandomForestClassifier(n_estimators=500, n_jobs=4)
model.fit(X, y)
print('Time:', time.time() - start)
