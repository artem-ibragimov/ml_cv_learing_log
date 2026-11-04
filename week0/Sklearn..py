import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
from pprint import pprint
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("./week0/possum.csv")
df = df.drop(columns=["case", "site", "Pop", "sex"]).dropna()

X = df.drop(["age"], axis=1)
y = df["age"]

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=44)


def euclodean_distance(vec1: np.ndarray, vec2: np.ndarray):
    return np.linalg.norm(vec1 - vec2)


def get_neighbors_numpy(
    train: np.ndarray,
    test_row: np.ndarray,
    num_neighbors: int,
):
    distances = np.linalg.norm(train - test_row, axis=1)
    nearest_neighbor_ids = distances.argsort()[:num_neighbors]
    return nearest_neighbor_ids


def predcit_numpy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    num_neighbors: int,
):
    y_predict = []

    for x_test in X_test:
        #   список id ближайших соседей для конкрентного x теста
        nearest_neighbor_ids = get_neighbors_numpy(X_train, x_test, num_neighbors)
        #   список y ближайших соседей
        nearest_neighbor_y = y_train[nearest_neighbor_ids]
        pred = np.mean(nearest_neighbor_y)
        y_predict.append(float(pred))

    return y_predict


y_predict = predcit_numpy(
    X_train=X_train[:30],
    X_test=X_test[:5],
    y_train=y_train[:30],
    num_neighbors=5,
)

print(y_predict)

model = DecisionTreeRegressor()
param_grid = {
    'max_depth': [1, 2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
}
gridsearch = GridSearchCV(model, param_grid=param_grid, refit=True, scoring=make_scorer(r2_score))

gridsearch.fit(X=X_train, y=y_train)
pprint(gridsearch.best_params_)

best_model = gridsearch.best_estimator_
print(f'MSE train = {mean_squared_error(y_train, best_model.predict(X_train))}')
print(f'MSE test = {mean_squared_error(y_test, best_model.predict(X_test))}')

import pickle

filename = 'best_model.sav'
pickle.dump(best_model, open(filename, 'wb'))
