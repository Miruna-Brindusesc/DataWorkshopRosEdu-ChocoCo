import pandas as pd

df = pd.read_csv("data/processed/cleaned_sales.csv")

print(df.head())
print(df.info())
print(df.describe())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

# Revenue by country
print(df.groupby("country")["amount"].sum().sort_values(ascending=False))

# Revenue by product
print(df.groupby("product")["amount"].sum().sort_values(ascending=False))

# Revenue by sales_person
print(df.groupby("sales_person")["amount"].sum().sort_values(ascending=False))