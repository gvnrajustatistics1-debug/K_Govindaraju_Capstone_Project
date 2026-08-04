pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib jupyter
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# Better display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

df = sns.load_dataset("titanic")
df.head()
df.to_csv("titanic.csv", index=False)

print("Titanic dataset saved successfully!")
print("Shape:", df.shape)
df.info()
df.describe(include="all")
df.head()
df.tail()
df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_percent = missing_percent[missing_percent > 0]

missing_percent.sort_values(ascending=False)
# Cleaning Strategy
#Missing <5% → Drop rows
#Missing between 5% and 30% → Impute values
# Missing >30% → Drop the column because excessive missing values reduce reliability.
df.drop(columns=["deck"], inplace=True)
df["age"].fillna(df["age"].median(), inplace=True)
df.dropna(subset=["embarked"], inplace=True)
df.dropna(subset=["embark_town"], inplace=True)
df.isnull().sum()
df.to_csv("titanic.csv", index=False)
print("Cleaned dataset saved.")
