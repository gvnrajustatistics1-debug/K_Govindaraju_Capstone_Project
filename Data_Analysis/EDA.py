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

# Histogram for Age

plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()
# Box Plot for Age
plt.figure(figsize=(8,4))

sns.boxplot(x=df["age"])

plt.title("Age Box Plot")

plt.show()
# Histogram for Fare
plt.figure(figsize=(10,5))

sns.histplot(df["fare"], bins=40, kde=True)

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Count")

plt.show()
#Box Plot for Fare
plt.figure(figsize=(8,4))

sns.boxplot(x=df["fare"])

plt.title("Fare Box Plot")

plt.show()
# IQR Outlier Detection
def iqr_outliers(column):

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"{column.upper()}")

    print("Q1 =", Q1)

    print("Q3 =", Q3)

    print("IQR =", IQR)

    print("Outliers =", len(outliers))

    return outliers
age_outliers = iqr_outliers("age")
fare_outliers = iqr_outliers("fare")
# Mean, Median and Mode of Fare
mean_fare = df["fare"].mean()

median_fare = df["fare"].median()

mode_fare = df["fare"].mode()[0]

print("Mean :", mean_fare)

print("Median :", median_fare)

print("Mode :", mode_fare)
# Determine Skewness
if mean_fare > median_fare > mode_fare:
    print("Fare is Right Skewed")

elif mean_fare < median_fare < mode_fare:
    print("Fare is Left Skewed")

else:
    print("Distribution is nearly Symmetric")

male_survival = df[df["sex"]=="male"]["survived"].mean()*100

female_survival = df[df["sex"]=="female"]["survived"].mean()*100

print(f"Male Survival Rate : {male_survival:.2f}%")

print(f"Female Survival Rate : {female_survival:.2f}%")
# Survival Rate by Passenger Class
for pclass in sorted(df["pclass"].unique()):

    rate = df[df["pclass"]==pclass]["survived"].mean()*100

    print(f"Class {pclass}: {rate:.2f}%")

# Survival Rate by Gender and Class
for gender in ["male","female"]:

    for pclass in [1,2,3]:

        rate = df[
            (df["sex"]==gender) &
            (df["pclass"]==pclass)
        ]["survived"].mean()*100

        print(f"{gender} | Class {pclass} | {rate:.2f}%")
# Notice the use of:
(df["sex"]=="female") & (df["pclass"]==1)
