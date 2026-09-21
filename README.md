# Experiment 3 — Regression Analysis Using Linear and Regularized Models

## ICS1512 — Machine Learning Algorithms Laboratory

This experiment implements and compares **Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net Regression** for predicting loan sanction amounts.

The notebook combines data cleaning, exploratory data analysis, preprocessing, hyperparameter tuning, 5-fold cross-validation, test-set evaluation, coefficient analysis, learning curves, and an overfitting/bias–variance analysis.

> **Important:** The uploaded PDF is titled **Experiment 2 — Email Spam/Ham Classification using Naïve Bayes and KNN**, while the uploaded notebook is **Experiment 3 — Regression Analysis using Linear and Regularized Models**. They describe different experiments and datasets. This README is therefore based primarily on the **uploaded `loan_regression_analysis.ipynb`**, with the PDF/report mismatch explicitly noted rather than combining unrelated results.

---

## 1. Objective

The notebook aims to:

- Predict **Loan Sanction Amount (USD)** from customer and loan-related attributes.
- Implement a baseline **Linear Regression** model.
- Implement **Ridge Regression** with L2 regularization.
- Implement **Lasso Regression** with L1 regularization.
- Implement **Elastic Net Regression** with combined L1/L2 regularization.
- Tune regularization hyperparameters using **GridSearchCV**.
- Evaluate models using:
  - MAE
  - MSE
  - RMSE
  - R²
- Validate model performance using **5-fold cross-validation**.
- Visualize predicted vs. actual values.
- Analyze residuals.
- Study learning curves for training and validation error.
- Compare learned coefficients across models.
- Examine overfitting, underfitting, regularization, and the bias–variance trade-off.

---

## 2. Dataset

### Dataset Name

**Predict Loan Amount Data**

The notebook expects a CSV file named:

```text
train.csv
```

and loads it from:

```text
dataset/train.csv
```

The notebook reports the initial dataset shape as:

```text
30,000 rows × 24 columns
```

The prediction target is:

```text
Loan Sanction Amount (USD)
```

### Dataset Source

The notebook identifies the source as Kaggle's **Predict Loan Amount Data** dataset.

The dataset contains customer, income, loan, property, employment, and financial attributes.

---

## 3. Target Variable

The target variable is:

```text
Loan Sanction Amount (USD)
```

This makes the task a **supervised regression problem**, because the output is a continuous numerical value.

---

## 4. Project Workflow

```text
                    Loan Dataset
                         |
                         v
                  Data Loading
                         |
                         v
                 Dataset Overview
                         |
                         v
                       EDA
              /          |          \
       Distribution   Scatter      Correlation
                         |
                         v
                  Data Preprocessing
                         |
          +--------------+--------------+
          |              |              |
     Remove IDs     Handle -999     Missing Values
          |              |              |
          +--------------+--------------+
                         |
                  Feature Separation
                  /                 \
             Numerical          Categorical
                  |                 |
            Median Impute      Mode Impute
            Standardize        One-Hot Encode
                  \                 /
                   +---------------+
                         |
                         v
                  Train/Test Split
                       80/20
                         |
          +--------------+--------------+
          |              |              |
       Linear          Ridge          Lasso
     Regression     Regression     Regression
          |              |              |
          +--------------+--------------+
                         |
                    Elastic Net
                         |
                         v
                GridSearchCV (5-Fold)
                         |
                         v
              Cross-Validation Results
                         |
                         v
                 Test Set Evaluation
                         |
          +--------------+--------------+
          |              |              |
     Predicted vs     Residuals     Learning Curves
       Actual
                         |
                         v
              Coefficient Comparison
                         |
                         v
              Bias–Variance Analysis
```

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Data loading and manipulation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, regression, tuning, validation, and metrics |

---

## 6. Data Preprocessing

The notebook performs the following preprocessing steps.

### 6.1 Remove Identifier Columns

The following columns are removed because they are treated as identifiers rather than predictive features:

```text
Customer ID
Name
Property ID
```

### 6.2 Handle Sentinel Values

The notebook treats:

```text
-999
```

as a missing/invalid value for numerical columns.

These values are replaced with `NaN`.

### 6.3 Remove Missing Target Rows

Rows where:

```text
Loan Sanction Amount (USD)
```

is missing are removed.

The resulting cleaned dataset has:

```text
29,322 rows × 21 columns
```

### 6.4 Numerical Features

Numerical columns are processed using:

```text
Median Imputation
        ↓
StandardScaler
```

Median imputation is used for missing numerical values.

### 6.5 Categorical Features

Categorical columns are processed using:

```text
Most-Frequent Imputation
        ↓
OneHotEncoder
```

The encoder uses:

```python
handle_unknown="ignore"
```

so unseen categories do not cause prediction errors.

### 6.6 Pipeline-Based Preprocessing

A `ColumnTransformer` combines the numerical and categorical preprocessing pipelines.

This preprocessing is included inside the Scikit-learn model pipelines, helping ensure that transformations are applied consistently during cross-validation and prediction.

---

## 7. Exploratory Data Analysis

The notebook performs several EDA visualizations.

### Target Distribution

A histogram with KDE is used to inspect the distribution of:

```text
Loan Sanction Amount (USD)
```

### Feature vs Target Relationships

Scatter plots are generated for:

- Income (USD)
- Loan Amount Request (USD)
- Credit Score
- Property Price

against the target.

### Correlation Analysis

A correlation heatmap is generated for numerical features and the target.

These visualizations help inspect:

- Distribution of the target
- Relationships between important predictors and loan sanction amount
- Linear relationships
- Correlation between numerical variables

---

## 8. Train/Test Split

The cleaned data is divided into:

```text
Training set: 23,457 samples
Test set:       5,865 samples
```

The split uses:

```python
test_size=0.2
random_state=42
```

The same held-out test set is used to compare the models.

---

## 9. Models Implemented

### 9.1 Linear Regression

Linear Regression is used as the baseline model.

It assumes that the target can be approximated as a linear combination of the input features:

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

---

### 9.2 Ridge Regression

Ridge Regression adds an L2 penalty to the ordinary least-squares objective.

The notebook tunes:

```text
alpha ∈ {0.01, 0.1, 1, 10, 100}
```

---

### 9.3 Lasso Regression

Lasso Regression applies an L1 penalty.

The notebook tunes:

```text
alpha ∈ {0.001, 0.01, 0.1, 1, 10}
```

Lasso can drive some coefficients to zero, providing an implicit form of feature selection.

---

### 9.4 Elastic Net Regression

Elastic Net combines L1 and L2 regularization.

The notebook tunes:

```text
alpha ∈ {0.01, 0.1, 1, 10}

l1_ratio ∈ {0.2, 0.5, 0.8}
```

---

## 10. Hyperparameter Tuning

The three regularized models are tuned using:

```text
GridSearchCV
```

with:

```text
5-fold cross-validation
```

and:

```text
random_state = 42
```

The scoring metric used during tuning is:

```text
R²
```

### Best Hyperparameters

| Model | Best Parameters | Best CV R² |
|---|---|---:|
| Ridge | alpha = 0.1 | 0.651146 |
| Lasso | alpha = 10 | 0.651563 |
| Elastic Net | alpha = 0.01, l1_ratio = 0.8 | 0.651060 |

These values are the results recorded in the executed notebook.

---

## 11. Evaluation Metrics

Four regression metrics are calculated.

### Mean Absolute Error — MAE

```text
MAE = average(|y - ŷ|)
```

Lower values indicate smaller absolute prediction errors.

### Mean Squared Error — MSE

```text
MSE = average((y - ŷ)²)
```

Larger errors receive greater penalty because the errors are squared.

### Root Mean Squared Error — RMSE

```text
RMSE = √MSE
```

RMSE is expressed in the same units as the target.

### R² Score

R² measures the proportion of variance in the target explained by the model.

A higher R² indicates better fit relative to the baseline mean prediction.

---

## 12. Cross-Validation Results

The notebook performs 5-fold cross-validation on the training data.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 18,903.09 | 8.1632e+08 | 28,571.34 | 0.651145 |
| Ridge Regression | 18,902.63 | 8.1632e+08 | 28,571.32 | 0.651146 |
| Lasso Regression | 18,882.84 | 8.1535e+08 | 28,554.32 | 0.651563 |
| Elastic Net Regression | 18,893.36 | 8.1653e+08 | 28,574.94 | 0.651060 |

The cross-validation results are very close across the four models.

---

## 13. Test Set Results

The final models are evaluated on the held-out test set.

| Model | MAE | MSE | RMSE | R² | Training Time (s) |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 18,761.75 | 7.7042e+08 | 27,756.40 | 0.657490 | 0.160 |
| Ridge Regression | 18,760.85 | 7.7038e+08 | 27,755.67 | 0.657508 | 5.889 |
| Lasso Regression | 18,744.35 | 7.7017e+08 | 27,751.91 | 0.657601 | 61.963 |
| Elastic Net Regression | 18,762.97 | 7.7071e+08 | 27,761.58 | 0.657362 | 54.190 |

The recorded test-set metrics are very similar across all four approaches.

---

## 14. Coefficient Analysis

The notebook extracts the transformed feature names from the preprocessing pipeline and compares coefficients across:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

The top 15 features are selected based on the absolute coefficient magnitude of Linear Regression.

Important observations from the executed notebook include:

- `Property Age` has a large negative coefficient in Linear and Ridge Regression.
- `Income (USD)` has a large positive coefficient in Linear and Ridge Regression.
- `Loan Amount Request (USD)` has a large positive coefficient.
- `Credit Score` also has a positive coefficient.
- Ridge generally shrinks coefficient magnitudes compared with Linear Regression.
- Lasso produces much stronger coefficient sparsity.

---

## 15. Lasso Feature Sparsity

The notebook reports:

```text
Lasso set 18 out of 53 coefficients to (near) zero
```

This demonstrates the feature-selection effect of L1 regularization.

Unlike Ridge, which primarily shrinks coefficients toward zero, Lasso can make coefficients exactly or approximately zero.

---

## 16. Overfitting and Underfitting Analysis

The notebook compares training and test R².

| Model | Train R² | Test R² | Gap |
|---|---:|---:|---:|
| Linear Regression | 0.661301 | 0.657490 | 0.003811 |
| Ridge Regression | 0.661300 | 0.657508 | 0.003792 |
| Lasso Regression | 0.661028 | 0.657601 | 0.003427 |
| Elastic Net Regression | 0.661058 | 0.657362 | 0.003696 |

The train-test gaps are small for all four models.

This indicates that, in the recorded experiment, there is **no large train-test performance gap** suggesting severe overfitting.

The notebook also compares training and validation error using learning curves.

---

## 17. Bias–Variance Analysis

The experiment uses regularization to study the bias–variance trade-off.

### Linear Regression

Linear Regression provides the unregularized baseline.

With many transformed features, it can have relatively low bias but may be more sensitive to variance.

### Ridge Regression

Ridge applies L2 regularization.

Increasing the regularization strength generally:

- Shrinks coefficients
- Reduces model variance
- Can increase bias

### Lasso Regression

Lasso applies L1 regularization.

It can:

- Shrink coefficients
- Set some coefficients to zero
- Produce a sparse model
- Perform implicit feature selection

### Elastic Net

Elastic Net combines:

- L1 regularization
- L2 regularization

This allows it to provide both coefficient sparsity and coefficient shrinkage.

---

## 18. Visualizations Generated

The notebook generates:

1. Loan sanction amount distribution
2. Feature-vs-target scatter plots
3. Numerical correlation heatmap
4. Test MAE comparison
5. Test RMSE comparison
6. Test R² comparison
7. Predicted vs actual plots for all models
8. Residual plots for all models
9. Learning curves
10. Coefficient comparison plot for the top 15 features

These visualizations provide complementary views of model performance rather than relying only on a single metric.

---

## 19. Project Structure

Recommended repository structure:

```text
loan-regression-analysis/
│
├── loan_regression_analysis.ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   └── train.csv
│
└── outputs/
    └── figures/
```

---

## 20. Installation

### 1. Clone or download the project

Place the notebook in the project directory.

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
loan_regression_analysis.ipynb
```

---

## 21. Dataset Setup

Download the required loan dataset and place the CSV at:

```text
dataset/train.csv
```

Alternatively, modify:

```python
DATA_PATH = "dataset/train.csv"
```

in the notebook to point to the actual dataset location.

---

## 22. Reproducing the Experiment

Run the notebook cells in order:

```text
1. Imports and Setup
2. Load Dataset
3. Dataset Overview
4. Data Preprocessing
5. Exploratory Data Analysis
6. Train/Test Split
7. Baseline Linear Regression
8. Regularized Models
9. 5-Fold Cross-Validation
10. Test Set Evaluation
11. Predicted vs Actual Plots
12. Residual Plots
13. Learning Curves
14. Coefficient Comparison
15. Overfitting Analysis
16. Bias–Variance Analysis
17. Final Summary Tables
18. Conclusion
```

---

## 23. Reproducibility

The notebook uses:

```python
RANDOM_STATE = 42
```

for reproducibility in the train/test split and K-fold configuration.

The K-fold cross-validation configuration is:

```text
5 folds
shuffle=True
random_state=42
```

---

## 24. Limitations

The following points should be considered when interpreting the experiment:

- The notebook depends on an external Kaggle dataset that is not included with the notebook.
- The dataset path is configured as `dataset/train.csv` and must exist before execution.
- The experiment evaluates four linear/regularized models; nonlinear models are outside its scope.
- The reported metrics depend on the particular dataset split and preprocessing configuration.
- The notebook's coefficient analysis is performed after one-hot encoding and standardization, so coefficient magnitudes should be interpreted in the context of the transformed feature space.
- The uploaded PDF report does **not** correspond to this notebook. The PDF describes a different Spambase classification experiment, so its results are intentionally not mixed into this regression README.

---

## 25. Key Findings

Based on the executed notebook:

- The dataset initially contained **30,000 rows and 24 columns**.
- After removing identifier columns, handling `-999` values, and removing rows with missing targets, **29,322 rows and 21 columns** remained.
- The data was split into **23,457 training samples and 5,865 test samples**.
- Grid search selected:
  - Ridge: `alpha = 0.1`
  - Lasso: `alpha = 10`
  - Elastic Net: `alpha = 0.01`, `l1_ratio = 0.8`
- The recorded test R² values were approximately **0.657** for all four models.
- Lasso produced **18 near-zero coefficients out of 53 transformed features**.
- The train-test R² gaps were small, ranging from approximately **0.0034 to 0.0038**.
- The regularized models required substantially more recorded training time than the baseline Linear Regression in this notebook.

---

## 26. Conclusion

This experiment demonstrates how Linear Regression and regularized regression methods can be applied to a real-world loan prediction problem.

The workflow covers the complete regression pipeline: data cleaning, numerical and categorical preprocessing, exploratory analysis, model construction, hyperparameter tuning, cross-validation, test evaluation, visualization, coefficient analysis, and bias–variance analysis.

The recorded results show very similar predictive performance among Linear Regression, Ridge, Lasso, and Elastic Net. Regularization mainly changes coefficient behaviour and model complexity in this experiment, with Lasso additionally producing a sparse coefficient representation.

The notebook therefore provides a practical demonstration of how regularization can be incorporated into a regression workflow and evaluated systematically rather than relying on a single train/test score.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** Regression Analysis Using Linear and Regularized Models
