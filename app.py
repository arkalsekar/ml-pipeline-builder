import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="No-Code ML Pipeline Builder",
    layout="wide"
)

st.title("🧩 No-Code Machine Learning Pipeline Builder")
st.caption("Build & run ML pipelines visually — no coding required")

# -------------------- SESSION STATE --------------------
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------- STEP 1: DATASET UPLOAD --------------------
st.header("1️⃣ Dataset Upload")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.success("Dataset uploaded successfully!")

        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Rows:**", df.shape[0])
            st.write("**Columns:**", df.shape[1])

        with col2:
            st.write("**Column Names:**")
            st.write(list(df.columns))

        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error reading file: {e}")

# -------------------- STEP 2: PREPROCESSING --------------------
if st.session_state.df is not None:
    st.header("2️⃣ Data Preprocessing")

    df = st.session_state.df

    target_column = st.selectbox(
        "Select Target Column (Label)",
        df.columns
    )

    feature_columns = df.drop(columns=[target_column]).columns

    preprocessing_option = st.radio(
        "Choose Preprocessing Method",
        ["None", "Standardization", "Normalization"]
    )

# -------------------- STEP 3: TRAIN-TEST SPLIT --------------------
if st.session_state.df is not None:
    st.header("3️⃣ Train–Test Split")

    split_ratio = st.slider(
        "Select Test Size (%)",
        min_value=20,
        max_value=40,
        value=30,
        step=5
    )

# -------------------- STEP 4: MODEL SELECTION --------------------
if st.session_state.df is not None:
    st.header("4️⃣ Model Selection")

    model_choice = st.selectbox(
        "Choose a Model",
        ["Logistic Regression", "Decision Tree Classifier"]
    )

# -------------------- STEP 5: TRAIN & RESULTS --------------------
if st.session_state.df is not None:
    st.header("5️⃣ Model Training & Results")

    if st.button("🚀 Run ML Pipeline"):

        X = df[feature_columns]
        y = df[target_column]

        # Apply preprocessing
        if preprocessing_option == "Standardization":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif preprocessing_option == "Normalization":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_ratio / 100,
            random_state=42
        )

        # Model selection
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = DecisionTreeClassifier()

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.success("✅ Model executed successfully!")

        st.metric("🎯 Accuracy", f"{accuracy:.2f}")

        # Confusion Matrix
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.caption("Pipeline Flow: Data → Preprocessing → Split → Model → Output")
