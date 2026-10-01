# Predict Employee Turnover

Predicts whether an employee will **leave or stay** in a company, using interpretable machine learning in Python. The project compares a Decision Tree, XGBoost and a hybrid DT-XGBoost model, then explains their predictions with **LIME**.

## Dataset

`Employee.csv` (not included in this repo) contains employee records with these columns: Education, JoiningYear, City, PaymentTier, Age, Gender, EverBenched, ExperienceInCurrentDomain and the target **LeaveOrNot**.

## Workflow

1. **Exploratory data analysis**: removed duplicate rows, checked for missing values, and plotted pie charts, box plots (to check for outliers), count plots and correlation heatmaps.
2. **Transformation**: encoded the categorical columns (Gender, City, EverBenched, Education) as numbers.
3. **Feature selection**: used `SelectKBest` with the ANOVA F-test and kept the four strongest features: **JoiningYear, PaymentTier, Age and Gender**.
4. **Modelling**: trained and evaluated three classifiers on an 80/20 train/test split.
5. **Explainability**: used LIME to show which features drive each model's prediction for an individual employee.

## Results

| Model | Accuracy |
|---|---|
| Decision Tree (entropy, max depth 8) | 73.4% |
| XGBoost | 73.1% |
| Hybrid DT-XGBoost (soft voting) | **74.0%** |

The hybrid model reached 0.76 precision, 0.50 recall and 0.60 F1 score.

## Files

- `Employees_future_prediction.ipynb`: the full notebook with outputs (originally run in Google Colab)
- `employees_future_prediction.py`: the same code exported as a Python script
- `Employee_Future_prediction.pdf`: the project report

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn, XGBoost, LIME
