import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/processed/cleaned_sales.csv")
df["date"] = pd.to_datetime(df["date"], errors = "coerce")

monthly = (
    df.groupby(df["date"].dt.to_period("M"))["amount"]
    .sum()
)

monthly.index = monthly.index.astype(str)

plt.plot(monthly.index, monthly.values)
plt.xticks(rotation=45)
plt.title("Monthly Revenue")
plt.tight_layout()
plt.savefig("reports/figures/monthly_revenue.png")
# plt.show()
plt.close()

top = df.groupby(df["product"])["amount"].sum().sort_values(ascending = False).head(10)
# top.index = top.index.astype(str)
top.plot(kind="barh", color="skyblue")
# plt.plot(top.index, top.values)
plt.title("Top 10 products by revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("reports/figures/top_10_products.png")
# plt.show()
print(df)
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(df["boxes_shipped"], df["amount"], alpha=0.7)
plt.xlabel("Boxes Shipped")
plt.ylabel("Revenue (Amount)")
plt.title("Boxes Shipped vs Revenue")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/figures/boxes_vs_revenue.png")
# plt.show()
plt.close()