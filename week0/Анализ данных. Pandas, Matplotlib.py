import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# array = np.random.randint(1, 100, size=(4, 4))
# print(array)
df = pd.read_csv("./week0/titanic.csv")
# df = df.dropna()
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\\.", expand=False)
df = df.drop(columns=["Cabin", "Name", "Ticket"])
df["Title"] = df["Title"].replace(
    [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ],
    "Rare",
)

df["Title"] = df["Title"].replace("Mlle", "Miss")
df["Title"] = df["Title"].replace("Ms", "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")

categorical_columns = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Title"]
categorical_features = df[categorical_columns]
unprocessed_cat_features = categorical_features.select_dtypes(
    include=[object]
).columns.tolist()

label_encoding = LabelEncoder()
for column in unprocessed_cat_features:
   df[column] = label_encoding.fit_transform(df[column])

print(df.head())

# print(df_encoded)
r