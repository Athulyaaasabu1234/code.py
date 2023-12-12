# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the automobile sales dataset
df = pd.read_csv("automobile_sales.csv")

# Compute summary statistics using numpy and pandas
summary_stats = df.describe()

# Set up the matplotlib figure and axes for subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("Automobile Sales Insights\nStudent Name: Athulya Sabu\nStudent ID: 22082059", fontsize=16)

# Plot 1: Bar plot of total sales per product line
sns.barplot(x="PRODUCTLINE", y="SALES", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Total Sales per Product Line")

# Tilt the x-axis labels
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Line plot of quantity ordered over time
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df['MonthYear'] = df['ORDERDATE'].dt.to_period('M')
monthly_quantity = df.groupby('MonthYear')['QUANTITYORDERED'].sum()
monthly_quantity.plot(ax=axes[0, 1], marker='o', linestyle='-')
axes[0, 1].set_title("Monthly Quantity Ordered Over Time")

# Add some gap between the top and bottom plots
fig.subplots_adjust(hspace=0.65)

# Plot 3: Box plot of deal size distribution
sns.boxplot(x="DEALSIZE", y="SALES", data=df, ax=axes[1, 0])
axes[1, 0].set_title("Deal Size Distribution")

# Plot 4: Count plot of order status
sns.countplot(x="STATUS", data=df, ax=axes[1, 1])
axes[1, 1].set_title("Order Status Count")

# Tilt the x-axis labels
axes[0, 0].tick_params(axis='x', rotation=45)

# Add explanation texts
axes[0, 0].text(0.5, -0.3, "Sales distribution across product lines", ha="center", transform=axes[0, 0].transAxes)
axes[0, 1].text(0.5, -0.3, "Monthly variation in quantity ordered", ha="center", transform=axes[0, 1].transAxes)
axes[1, 0].text(0.5, -0.3, "Sales distribution based on deal size", ha="center", transform=axes[1, 0].transAxes)
axes[1, 1].text(0.5, -0.3, "Distribution of order statuses", ha="center", transform=axes[1, 1].transAxes)

# Save the infographic as a PNG file
plt.savefig("22082059.png", dpi=300)
