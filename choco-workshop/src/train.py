import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ridge = Ridge(alpha=1.0)  

df = pd.read_csv("data/processed/cleaned_sales.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Feature engineering
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"] >= 5

categorical_cols = ["country", "product", "sales_person"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(columns=["amount", "date"])
y = df["amount"]

# Train 80%
df_sorted = df.sort_values("date")
train_size = int(0.8 * len(df_sorted))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

# Test 20%
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

# Mean baseline
y_pred_mean = np.full_like(y_test, y_train.mean())
mae_mean = mean_absolute_error(y_test, y_pred_mean)
print("Baseline MAE (mean):", mae_mean)

# Median baseline
y_pred_median = np.full_like(y_test, y_train.median())
mae_median = mean_absolute_error(y_test, y_pred_median)
print("Baseline MAE (median):", mae_median)

# Mean baseline
y_pred_mean = np.full_like(y_test, y_train.mean())
print("MAE baseline (mean):", mean_absolute_error(y_test, y_pred_mean))

# Evaluation
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(ridge, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
print("CV MAE scores:", -scores)