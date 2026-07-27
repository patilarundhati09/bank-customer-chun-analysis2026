import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("Bank customer churn prediction.csv")

# =====================================
# Dataset Preview
# =====================================
print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# =====================================
# Data Cleaning
# =====================================
columns_to_drop = ["customer_id"]

for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================
# Churn Statistics
# =====================================
print("\nCustomer Churn Statistics:")
churn_stats = df["churn"].value_counts()
print(churn_stats)

plt.figure(figsize=(6,5))
churn_stats.plot(kind="bar", color=["green", "red"])
plt.title("Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks([0,1], ["No", "Yes"], rotation=0)
plt.tight_layout()
plt.show()

# =====================================
# Gender-wise Churn
# =====================================
gender_churn = df.groupby("gender")["churn"].mean()

print("\nGender-wise Churn Rate:")
print(gender_churn)

plt.figure(figsize=(6,5))
gender_churn.plot(kind="bar", color="orange")
plt.title("Gender-wise Churn Rate")
plt.xlabel("Gender")
plt.ylabel("Average Churn Rate")
plt.tight_layout()
plt.show()

# =====================================
# Average Balance
# =====================================
balance_analysis = df.groupby("churn")["balance"].mean()

print("\nAverage Balance:")
print(balance_analysis)

plt.figure(figsize=(6,5))
balance_analysis.plot(kind="bar", color="blue")
plt.title("Average Balance")
plt.xlabel("Churn")
plt.ylabel("Average Balance")
plt.xticks([0,1], ["No", "Yes"], rotation=0)
plt.tight_layout()
plt.show()

# =====================================
# Age Distribution
# =====================================
plt.figure(figsize=(8,5))
plt.hist(df["age"], bins=20, color="purple", edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# =====================================
# Credit Score Distribution
# =====================================
plt.figure(figsize=(8,5))
plt.hist(df["credit_score"], bins=20, color="skyblue", edgecolor="black")
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# =====================================
# Dashboard
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(16,10))
fig.suptitle("Bank Customer Churn Dashboard", fontsize=18)

# Churn Statistics
churn_stats.plot(kind="bar", color=["green","red"], ax=axes[0,0])
axes[0,0].set_title("Customer Churn")
axes[0,0].set_xticklabels(["No","Yes"], rotation=0)

# Gender-wise Churn
gender_churn.plot(kind="bar", color="orange", ax=axes[0,1])
axes[0,1].set_title("Gender-wise Churn")

# Average Balance
balance_analysis.plot(kind="bar", color="blue", ax=axes[1,0])
axes[1,0].set_title("Average Balance")
axes[1,0].set_xticklabels(["No","Yes"], rotation=0)

# Age Distribution
axes[1,1].hist(df["age"], bins=20, color="purple", edgecolor="black")
axes[1,1].set_title("Age Distribution")
axes[1,1].set_xlabel("Age")
axes[1,1].set_ylabel("Customers")

plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

print("\nBank Customer Churn Analysis Completed Successfully!")
