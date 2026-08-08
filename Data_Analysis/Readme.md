# Titanic Data Analysis (Part A – Exploratory Data Analysis)

## Objective

The objective of this module is to perform a complete Exploratory Data Analysis (EDA) on the Titanic dataset. The dataset is loaded only once using Seaborn's built-in dataset loader and immediately saved as an offline CSV (`titanic.csv`). All subsequent analysis is performed using this saved dataset to ensure reproducibility without requiring an internet connection.

---

# Dataset

* **Source:** Seaborn Titanic Dataset
* **Loading Method:** `sns.load_dataset("titanic")`
* **Offline Backup:** `titanic.csv`

The dataset was loaded only once and saved as:

```python
df.to_csv("titanic.csv", index=False)
```

This file serves as the offline fallback for grading.

---

# Dataset Profiling

The following exploratory functions were performed:

* Dataset Shape
* Dataset Information (`df.info()`)
* Statistical Summary (`df.describe()`)
* First and Last Records
* Missing Value Count
* Missing Value Percentage

Missing-value percentages were calculated using:

```python
(df.isnull().sum() / len(df)) * 100
```

---

# Missing Value Handling Strategy

The following rule was applied consistently throughout the project.

| Missing Percentage | Strategy              |
| ------------------ | --------------------- |
| Less than 5%       | Drop affected rows    |
| Between 5% and 30% | Impute missing values |
| Greater than 30%   | Drop the column       |

### Column-wise Decisions

**Age**

* Missing values: Approximately 20%
* Strategy: Median Imputation
* Reason: Age is an important predictor and the missing percentage falls within the 5%–30% threshold.

**Embarked**

* Missing values: Less than 1%
* Strategy: Drop affected rows
* Reason: Very few records were missing.

**Embark Town**

* Missing values: Less than 1%
* Strategy: Drop affected rows
* Reason: The missing percentage is negligible.

**Deck**

* Missing values: Approximately 77%
* Strategy: Drop Column
* Reason: The missing percentage is too high for reliable imputation.

---

# Univariate Analysis

The following analyses were performed.

## Age

* Histogram
* Box Plot
* IQR-based Outlier Detection

## Fare

* Histogram
* Box Plot
* IQR-based Outlier Detection

---

# Outlier Detection

Outliers were identified using the Interquartile Range (IQR) rule.

Lower Bound

```
Q1 − 1.5 × IQR
```

Upper Bound

```
Q3 + 1.5 × IQR
```

The total number of outliers was calculated separately for:

* Age
* Fare

---

# Central Tendency

For the Fare column, the following statistics were calculated.

* Mean
* Median
* Mode

The relationship between these values was used to determine the distribution shape.

If

```
Mean > Median > Mode
```

then the distribution was concluded to be **positively (right) skewed**.

---

# Bivariate Analysis

Survival rates were computed using Boolean Masking for:

* Survival by Gender
* Survival by Passenger Class
* Survival by Gender and Passenger Class

Example:

```python
(df["sex"] == "female") & (df["pclass"] == 1)
```

These analyses help understand how gender and travel class influenced survival.

---

# Correlation Analysis

The correlation matrix was computed using exactly the following six columns:

* survived
* pclass
* age
* sibsp
* parch
* fare

The following columns were intentionally excluded:

* adult_male
* alone

These variables are derived features rather than independent measurements.

A heatmap was created using Seaborn to visualize feature relationships.

The two strongest feature correlations (based on absolute correlation values) were identified and interpreted.

---

# Multivariate Analysis

The following visualizations were created to understand survival patterns.

1. Survival Rate by Gender
2. Survival Rate by Passenger Class
3. Age vs Survival (Box Plot)
4. Age vs Fare (Scatter Plot)

Each visualization includes a written interpretation explaining its significance.

---

# Feature Standardization

As an exploratory analysis, the following numerical columns were standardized.

* Age
* Fare

Standardization was performed using the Z-score formula.

```
z = (x − mean) / standard deviation
```

A before-and-after comparison confirmed that the transformed features have approximately:

* Mean = 0
* Standard Deviation = 1

This standardization step was performed only for exploratory purposes and is not used in the predictive modeling pipeline.

---

# Key Findings

* Female passengers had a significantly higher survival rate than male passengers.
* First-class passengers had the highest survival probability.
* Fare distribution is positively skewed because a small number of passengers paid exceptionally high ticket prices.
* Fare contains substantially more outliers than Age.
* Passenger class and fare are strongly correlated.
* Family-related variables (SibSp and Parch) also show meaningful relationships.

---

# Files Included

```
analytics/
│
├── 01_eda.ipynb
├── titanic.csv
├── README.md
├── images/
└── outputs/
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# Conclusion

This exploratory data analysis provides a comprehensive understanding of the Titanic dataset by profiling the data, handling missing values, detecting outliers, analyzing feature relationships, and identifying factors associated with passenger survival. The cleaned dataset (`titanic.csv`) produced during this phase serves as the foundation for the predictive modeling pipeline implemented in Part B.
# Titanic Predictive Modeling (Part B)

## Objective

The objective of Part B is to build and evaluate machine learning models for predicting passenger survival on the Titanic dataset. The cleaned dataset (`titanic.csv`) generated in Part A is used throughout this stage. All preprocessing is performed using a Scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage and ensure that the trained model can be deployed on raw, unseen data.

---

# Dataset

**Dataset Used:** Titanic Dataset

**Input File:** `titanic.csv`

The dataset used in this phase is the cleaned dataset produced during Part A. No additional calls to `sns.load_dataset("titanic")` were made. This ensures that the dataset is loaded from the network only once across the entire project, satisfying the assignment requirement.

---

# Target Variable

The classification target is:

* **survived**

Target values:

* 0 → Not Survived
* 1 → Survived

---

# Features Used

The following features were selected for classification:

* Passenger Class (`pclass`)
* Sex (`sex`)
* Age (`age`)
* Number of Siblings/Spouses (`sibsp`)
* Number of Parents/Children (`parch`)
* Fare (`fare`)
* Port of Embarkation (`embarked`)

---

# Train-Test Split

The dataset was divided into training and testing sets using a **Stratified Train-Test Split**.

```python
train_test_split(
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

### Why Stratification?

The Titanic dataset is moderately imbalanced, with more passengers not surviving than surviving. Stratified sampling preserves the same class distribution in both the training and testing datasets, resulting in a fairer and more reliable model evaluation.

---

# Data Preprocessing

All preprocessing steps were fitted **only on the training dataset** and then applied to the test dataset using the trained transformers. This prevents information leakage.

The preprocessing pipeline includes:

### Numeric Features

* Median Imputation for missing values
* StandardScaler for feature scaling

### Categorical Features

* Most Frequent Imputation
* One-Hot Encoding

The preprocessing was implemented using:

* `ColumnTransformer`
* `Pipeline`

---

# Classification Models

Three machine learning classification algorithms were trained on the same train-test split.

## 1. Logistic Regression

A baseline linear classification model used for comparison.

Evaluation Metrics:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* Area Under Curve (AUC)

---

## 2. Decision Tree

A Decision Tree classifier was trained using the same preprocessing pipeline.

Additionally,

* Decision Tree visualization was created using `plot_tree()`.
* Feature names and class names were displayed.

Evaluation Metrics:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* Area Under Curve (AUC)

---

## 3. Random Forest

A Random Forest classifier was trained using the same train-test split and preprocessing pipeline.

Evaluation Metrics:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* Area Under Curve (AUC)

---

# Model Comparison

The performance of all three classifiers was summarized using the following metrics.

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The results were presented in a comparison table for easy evaluation.

---

# Imbalanced Data Handling

The class distribution was examined before model training.

Three different approaches were compared.

### 1. Baseline Model

No imbalance handling.

### 2. Class Weight

`class_weight="balanced"` was applied to Logistic Regression.

### 3. SMOTE

Synthetic Minority Oversampling Technique (SMOTE) was applied **only to the training dataset**.

Performance was compared using:

* Precision
* Recall
* F1 Score

This ensured that no information from the test dataset leaked into the training process.

---

# Hyperparameter Tuning

Random Forest hyperparameters were optimized using `GridSearchCV`.

The following parameters were tuned:

* `n_estimators`
* `max_depth`
* `max_features`

The Random Forest model was initialized with:

* `bootstrap=True`
* `oob_score=True`

The following were reported:

* Best Parameter Combination
* Best Cross-Validation Score
* Out-of-Bag (OOB) Score

---

# Regression Side Task

A multivariate Linear Regression model was developed to predict passenger fare.

Target Variable:

* Fare

Evaluation Metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Adjusted R² Score

A residual plot was generated to examine the regression assumptions.

### Residual Interpretation

If residuals are randomly distributed around zero, the model satisfies the constant variance assumption. A funnel-shaped pattern indicates heteroscedasticity.

---

# Model Persistence

The complete trained pipeline, including preprocessing and the final estimator, was saved using `joblib`.

```python
joblib.dump(best_pipeline, "best_pipeline.joblib")
```

The saved pipeline includes:

* Missing Value Imputation
* One-Hot Encoding
* Feature Scaling
* Final Classification Model

The pipeline was reloaded using:

```python
joblib.load("best_pipeline.joblib")
```

Predictions were successfully generated on raw, unprocessed input data, confirming that the complete pipeline was saved correctly.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib

---

# Files Included

```text
analytics/
│
├── 02_modeling.ipynb
├── titanic.csv
├── best_pipeline.joblib
├── README.md
├── outputs/
│   ├── classification_metrics.csv
│   ├── regression_metrics.csv
│   └── model_comparison.csv
└── images/
```

---

# Final Recommendation

Among the three classification models, the **Random Forest Classifier** demonstrated the best overall performance based on Accuracy, Precision, Recall, F1 Score, and ROC-AUC. Compared with Logistic Regression and Decision Tree, Random Forest handled nonlinear relationships more effectively and produced more stable predictions.

The comparison between the baseline model, class-weight balancing, and SMOTE showed that handling class imbalance improved minority-class prediction, particularly in Recall and F1 Score. SMOTE was applied only to the training data, preventing information leakage and ensuring a valid evaluation.

The complete preprocessing workflow and trained classifier were saved together as a single Scikit-learn Pipeline using Joblib. This allows the saved model to accept raw input data directly and perform preprocessing and prediction automatically, making it suitable for deployment in real-world applications.
