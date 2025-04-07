import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go

def run_diabetes_prediction_app():
    """
    Loads or generates data to predict diabetes risk using a Random Forest model.
    Includes train/test split, evaluation, and single-sample prediction.
    """
    st.title("Diabetes Prediction")
    st.markdown("""
    This section combines symptom/activity data to predict diabetes risk 
    using a simple Random Forest model.
    """)

    # Option: Upload your own data or use synthetic data
    uploaded_file = st.file_uploader("Upload CSV with features + 'Class'", type=["csv"])
    use_synthetic = st.checkbox("Use synthetic data instead")

    if not uploaded_file and not use_synthetic:
        st.info("Upload a CSV or select 'Use synthetic data' to continue.")
        return

    # Load or generate data
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded from your CSV!")
    else:
        np.random.seed(42)
        size = 200
        df = pd.DataFrame({
            "Age": np.random.randint(20, 80, size=size),
            "Gender": np.random.choice([0, 1], size=size), # 0 = Female, 1 = Male
            "Polyuria": np.random.choice([0, 1], size=size),
            "Polydipsia": np.random.choice([0, 1], size=size),
            "Heart_Rate": np.random.randint(60, 120, size=size),
            "Steps": np.random.randint(1000, 15000, size=size),
            "Glucose": np.random.randint(70, 200, size=size),
            "Sleep_Hours": np.round(np.random.uniform(4, 9, size=size),1),
            "Class": np.random.choice([0, 1], size=size, p=[0.7, 0.3])
        })
        st.warning("Using synthetic data for demonstration.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    # Confirm dataset includes 'Class' column
    if "Class" not in df.columns:
        st.error("Dataset must include a 'Class' column (0=Negative, 1=Positive).")
        return

    # Basic train/test split
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    st.write(f"**Model Accuracy**: {acc*100:.2f}%")

    st.subheader("Confusion Matrix")
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: 0", "Predicted: 1"],
        y=["Actual: 0", "Actual: 1"],
        colorscale="Blues"
    ))
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report")
    st.json(cr)

    # Single Prediction
    st.subheader("Try a Single Prediction")
    with st.form("diabetes_prediction_form"):
        input_age = st.number_input("Age", min_value=10, max_value=120, value=50)
        input_gender = st.selectbox("Gender", ["Female", "Male"])
        input_polyuria = st.selectbox("Polyuria (excess urination)?", ["No", "Yes"])
        input_polydipsia = st.selectbox("Polydipsia (excess thirst)?", ["No", "Yes"])
        input_hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)
        input_steps = st.number_input("Steps/day", min_value=0, max_value=50000, value=5000)
        input_glu = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=100)
        input_sleep = st.number_input("Sleep (hrs)", min_value=0.0, max_value=24.0, value=6.0)

        submitted = st.form_submit_button("Predict Diabetes Risk")
        if submitted:
            gender_map = {"Female":0, "Male":1}
            yes_no_map = {"No":0, "Yes":1}

            new_sample = pd.DataFrame([{
                "Age": input_age,
                "Gender": gender_map[input_gender],
                "Polyuria": yes_no_map[input_polyuria],
                "Polydipsia": yes_no_map[input_polydipsia],
                "Heart_Rate": input_hr,
                "Steps": input_steps,
                "Glucose": input_glu,
                "Sleep_Hours": input_sleep
            }])

            pred = model.predict(new_sample)[0]
            proba = model.predict_proba(new_sample)[0][pred]

            if pred == 1:
                st.error(f"High Diabetes Risk. Confidence: {proba*100:.2f}%")
            else:
                st.success(f"Low Diabetes Risk. Confidence: {proba*100:.2f}%")
