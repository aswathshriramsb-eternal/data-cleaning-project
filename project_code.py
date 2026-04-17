import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("student_data.csv")

print("Original Data:\n", df)

# Cleaning
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# Feature
df['Average'] = df[['Math','Science','English']].mean(axis=1)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

# Graphs
df[['Math','Science','English']].mean().plot(kind='bar')
plt.title("Average Marks")
plt.savefig("bar_chart.png")
plt.close()

df[['Math','Science','English']].plot()
plt.title("Marks Trend")
plt.savefig("line_chart.png")
plt.close()

sns.heatmap(df.select_dtypes(include='number').corr(), annot=True)
plt.title("Heatmap")
plt.savefig("heatmap.png")
plt.close()

print("Done ✅")