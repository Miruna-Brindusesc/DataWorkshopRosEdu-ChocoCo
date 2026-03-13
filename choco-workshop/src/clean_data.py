import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/sales.csv")

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Parse date
df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
print("Failed date parses for datetime:", df["date"].isna().sum())

# Convert numeric columns
df["amount"] = (
    df["amount"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "")
)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
print("Failed date parses for amount:", df["amount"].isna().sum())

df["boxes_shipped"] = pd.to_numeric(df["boxes_shipped"], errors="coerce")
print("Failed date parses for boxes shipped:", df["boxes_shipped"].isna().sum())

# Drop rows with missing target
before = len(df)
df = df.dropna(subset=["amount"])
after = len(df)

print("Rows removed due to missing Amount:", before - after)

# # Remove duplicates
df = df.drop_duplicates()
print("Duplicated data: ", df.duplicated().sum())

# Save cleaned dataset
df.to_csv("data/processed/cleaned_sales.csv", index=False)

print("Cleaning complete.")