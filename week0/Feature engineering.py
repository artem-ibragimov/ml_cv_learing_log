from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('./week0/bank-issues-data/train.csv')
test_data = pd.read_csv('./week0/bank-issues-data/test.csv')

# print(train_data.head())

y_train = train_data['open_account_flg']
train_data = train_data.drop(columns=['open_account_flg'])

from sklearn.impute import KNNImputer

# simple_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# simple_imputer.fit(train_data[['credit_count']])
# train_data['credit_count'] = simple_imputer.transform(train_data[['credit_count']])
# pprint([train_data[column].dtype for column in train_data.columns])
numeric_columns = [
    column for column in train_data.columns
    if train_data[column].dtype == np.dtype('float64') or train_data[column].dtype == np.dtype('int64')
]
# print(numeric_columns[1:])
# numeric_data = train_data[numeric_columns].drop(columns=['tariff_id', 'client_id'])
# print(numeric_data.columns)
# knn_imputer = KNNImputer(n_neighbors=2)
# knn_imputer.fit(numeric_data)
# transformed = pd.DataFrame(knn_imputer.transform(numeric_data), columns=numeric_data.columns)
# print(transformed['overdue_credit_count'].isna().any())

# Обработка тестовых данных

categorical_columns = train_data.select_dtypes(include=['object', 'string']).columns.tolist()
categorical_columns = [x for x in categorical_columns if x != 'living_region']
print(np.unique(train_data['marital_status']))

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
ohe.fit(train_data[categorical_columns])
new_category_columns = ohe.transform(train_data[categorical_columns])
pprint(ohe.get_feature_names_out())
