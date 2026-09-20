"""
Customer Analytics Using Python — Superstore Sales Dataset
Author: David Adebowale

What this script does (in order):
1. Load and clean the raw sales data
2. Exploratory data analysis (EDA) — sales by category, region, trend over time
3. RFM analysis (Recency, Frequency, Monetary) to segment customers
4. Save charts and a summary of findings

Dataset: "Sample Superstore" — a well-known public retail dataset,
9,994 real order line items from a US-based superstore.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------- 1. LOAD & CLEAN ----------
df = pd.read_csv("superstore.csv", encoding="latin1")

# Parse dates properly (they arrive as text)
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")

# Check for missing values and duplicates
missing = df.isnull().sum().sum()
duplicates = df.duplicated().sum()
df = df.drop_duplicates()

print(f"Rows after cleaning: {len(df)}")
print(f"Missing values found: {missing}")
print(f"Duplicate rows removed: {duplicates}")

# ---------- 2. EDA ----------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()
avg_order_value = df.groupby("Order ID")["Sales"].sum().mean()

print(f"\nTotal sales: ${total_sales:,.0f}")
print(f"Total profit: ${total_profit:,.0f}")
print(f"Total orders: {total_orders:,}")
print(f"Unique customers: {total_customers:,}")
print(f"Average order value: ${avg_order_value:,.2f}")

# Sales by category
cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)

# Sales by region
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

# Monthly sales trend
df["Month"] = df["Order Date"].dt.to_period("M")
monthly_sales = df.groupby("Month")["Sales"].sum()

# Top 5 and bottom 5 sub-categories by profit (interesting for a real business finding)
subcat_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()
worst_subcats = subcat_profit.head(5)
best_subcats = subcat_profit.tail(5)

# ---------- 3. RFM ANALYSIS ----------
snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Order ID", "nunique"),
    Monetary=("Sales", "sum"),
).reset_index()

# Score each dimension 1-4 using quartiles (4 = best)
rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

def segment(row):
    if row["RFM_score"] >= 10:
        return "Champions"
    elif row["RFM_score"] >= 8:
        return "Loyal Customers"
    elif row["RFM_score"] >= 6:
        return "Potential Loyalists"
    elif row["RFM_score"] >= 4:
        return "At Risk"
    else:
        return "Lost / Low Value"

rfm["Segment"] = rfm.apply(segment, axis=1)
segment_counts = rfm["Segment"].value_counts()
segment_value = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)

print("\nCustomer segments (count):")
print(segment_counts)
print("\nRevenue by segment:")
print(segment_value)

rfm.to_csv("rfm_table.csv", index=False)

# ---------- 4. CHARTS ----------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
NAVY = "#2A4365"
TEAL = "#3E8E7E"
GREY = "#8A93A0"

# Chart 1: Monthly sales trend
fig, ax = plt.subplots(figsize=(9, 4))
monthly_sales.plot(ax=ax, color=NAVY, linewidth=2)
ax.set_title("Monthly Sales Trend")
ax.set_ylabel("Sales ($)")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
plt.tight_layout()
plt.savefig("chart_monthly_trend.png", dpi=150)
plt.close()

# Chart 2: Sales by category
fig, ax = plt.subplots(figsize=(6, 4))
cat_sales.plot(kind="barh", ax=ax, color=TEAL)
ax.set_title("Sales by Category")
ax.set_xlabel("Sales ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
plt.tight_layout()
plt.savefig("chart_category_sales.png", dpi=150)
plt.close()

# Chart 3: Best and worst sub-categories by profit
fig, ax = plt.subplots(figsize=(7, 5))
combined = pd.concat([worst_subcats, best_subcats])
colors = ["#C0524A" if v < 0 else TEAL for v in combined.values]
combined.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Least & Most Profitable Sub-Categories")
ax.set_xlabel("Profit ($)")
ax.axvline(0, color="#333", linewidth=0.8)
plt.tight_layout()
plt.savefig("chart_subcategory_profit.png", dpi=150)
plt.close()

# Chart 4: Customer segments
fig, ax = plt.subplots(figsize=(6, 4))
order = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost / Low Value"]
segment_counts.reindex(order).plot(kind="bar", ax=ax, color=NAVY)
ax.set_title("Customers by RFM Segment")
ax.set_ylabel("Number of customers")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("chart_rfm_segments.png", dpi=150)
plt.close()

print("\nCharts saved. Analysis complete.")
