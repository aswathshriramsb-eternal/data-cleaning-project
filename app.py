import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Data Cleaning & Visualization Dashboard")

# Load data
try:
    raw_df = pd.read_csv("Student_data.csv")
    clean_df = pd.read_csv("cleaned_data.csv")
    st.success("Data loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Show raw data
st.subheader("📂 Raw Data")
st.dataframe(raw_df)

# Show cleaned data
st.subheader("🧹 Cleaned Data")
st.dataframe(clean_df)

# Compare missing values
st.subheader("⚠️ Missing Values Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("Raw Data Missing Values")
    st.write(raw_df.isnull().sum())

with col2:
    st.write("Cleaned Data Missing Values")
    st.write(clean_df.isnull().sum())

# Charts
st.subheader("📊 Visualizations (Cleaned Data)")

# Bar chart
fig1, ax1 = plt.subplots()
clean_df[['Math','Science','English']].mean().plot(kind='bar', ax=ax1)
ax1.set_title("Average Marks")
st.pyplot(fig1)

# Line chart
fig2, ax2 = plt.subplots()
clean_df[['Math','Science','English']].plot(ax=ax2)
ax2.set_title("Marks Trend")
st.pyplot(fig2)

# Heatmap
fig3, ax3 = plt.subplots()
sns.heatmap(clean_df[['Math','Science','English']].corr(), annot=True, ax=ax3)
ax3.set_title("Correlation Heatmap")
st.pyplot(fig3)

# Top performer
st.subheader("🏆 Top Performer")

clean_df['Average'] = clean_df[['Math','Science','English']].mean(axis=1)
top = clean_df.loc[clean_df['Average'].idxmax()]

st.success(f"Top Student: {top['Name']} with Average {round(top['Average'],2)}")