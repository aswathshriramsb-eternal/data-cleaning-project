
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 📌 LOAD DATASET
# ==============================

df = pd.read_csv("data.csv")

# ==============================
# 📌 BASIC INFO
# ==============================
print("===== BASIC INFORMATION =====")
print(df.info())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())

# ==============================
# 📌 DATA CLEANING
# ==============================
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Fill missing values (numerical with mean)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# ==============================
# 📌 STATISTICAL SUMMARY
# ==============================
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# ==============================
# 📌 UNIVARIATE ANALYSIS
# ==============================
print("\n===== UNIVARIATE ANALYSIS =====")

for col in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# ==============================
# 📌 CORRELATION ANALYSIS
# ==============================
print("\n===== CORRELATION MATRIX =====")

corr = df.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 📌 BIVARIATE ANALYSIS
# ==============================
print("\n===== BIVARIATE ANALYSIS =====")

columns = df.select_dtypes(include=np.number).columns

for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        plt.figure()
        sns.scatterplot(x=df[columns[i]], y=df[columns[j]])
        plt.title(f"{columns[i]} vs {columns[j]}")
        plt.show()

# ==============================
# 📌 KEY INSIGHTS (AUTO)
# ==============================
print("\n===== KEY INSIGHTS =====")

# Strong correlations
strong_corr = corr.unstack().sort_values(ascending=False)

print("\nTop Correlations:")
print(strong_corr[(strong_corr < 1) & (strong_corr > 0.7)].head())

# Negative correlations
print("\nStrong Negative Correlations:")
print(strong_corr[strong_corr < -0.7].head())

# ==============================
# 📌 OUTLIER DETECTION
# ==============================
print("\n===== OUTLIER DETECTION =====")

for col in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# ==============================
# 📌 FINAL REPORT SUMMARY
# ==============================
print("\n===== FINAL REPORT =====")

print("""
1. Dataset cleaned (missing values handled, duplicates removed)
2. Statistical summary generated
3. Distribution of variables analyzed
4. Correlation between variables identified
5. Strong influencing features detected
6. Outliers visualized
7. Patterns and trends observed using graphs
""")