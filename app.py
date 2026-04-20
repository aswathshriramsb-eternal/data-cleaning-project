# ==============================
# 📌 IMPORT LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ==============================
# 📌 STREAMLIT UI
# ==============================
st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊 Exploratory Data Analysis + ML Dashboard")

# ==============================
# 📌 FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    # ==============================
    # 📌 DATA CLEANING
    # ==============================
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    # ==============================
    # 📌 STATISTICS
    # ==============================
    st.subheader("📊 Statistical Summary")
    st.write(df.describe())

    # ==============================
    # 📌 CORRELATION HEATMAP
    # ==============================
    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ==============================
    # 📌 FEATURE SELECTION
    # ==============================
    st.subheader("🎯 Select Target Column")

    target = st.selectbox("Choose Target Column", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical target
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Keep only numeric features
        X = X.select_dtypes(include=np.number)

        # ==============================
        # 📌 TRAIN TEST SPLIT
        # ==============================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ==============================
        # 📌 MODEL TRAINING
        # ==============================
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # ==============================
        # 📌 PREDICTIONS
        # ==============================
        y_pred = model.predict(X_test)

        st.subheader("📈 Model Performance")

        # ==============================
        # 📌 CONFUSION MATRIX
        # ==============================
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # ==============================
        # 📌 CLASSIFICATION REPORT
        # ==============================
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # ==============================
        # 📌 ROC CURVE
        # ==============================
        if len(np.unique(y)) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle='--')
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)

        # ==============================
        # 📌 FEATURE IMPORTANCE
        # ==============================
        st.subheader("⭐ Feature Importance")

        importance = pd.Series(model.coef_[0], index=X.columns)
        importance = importance.sort_values(ascending=False)

        st.bar_chart(importance)

        # ==============================
        # 📌 INSIGHTS
        # ==============================
        st.subheader("🧠 Key Insights")

        st.write("Top Influencing Features:")
        st.write(importance.head())

        st.write("Least Influencing Features:")
        st.write(importance.tail())

else:
    st.info("👆 Upload a dataset to begin")