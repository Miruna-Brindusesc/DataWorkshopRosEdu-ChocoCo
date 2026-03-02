import pandas as pd

df = pd.read_csv("data/processed/cleaned_sales.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Extra date features
df["month"] = df["date"].dt.month           # luna (1-12)
df["day_of_week"] = df["date"].dt.dayofweek # ziua săptămânii (0=Luni, 6=Duminică)
df["is_weekend"] = df["day_of_week"] >= 5   # True pentru Sâmbătă/Duminică

# Categorical variables
df = pd.get_dummies(df, columns=["country","product","sales_person"], drop_first=True)

# Target
X = df.drop(columns=["amount","date"])
y = df["amount"]