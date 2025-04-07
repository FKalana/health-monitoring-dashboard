import streamlit as st

# Import the other three apps
from activity_monitoring_app import run_activity_monitoring_app
from health_dashboard_app import run_health_dashboard_app
from diabetes_prediction_app import run_diabetes_prediction_app

def main():
    st.sidebar.title("Main Menu")
    selection = st.sidebar.radio(
        "Go to",
        ("Activity Monitoring", "Health Monitoring Dashboard", "Diabetes Prediction")
    )

    if selection == "Activity Monitoring":
        run_activity_monitoring_app()
    elif selection == "Health Monitoring Dashboard":
        run_health_dashboard_app()
    elif selection == "Diabetes Prediction":
        run_diabetes_prediction_app()

if __name__ == "__main__":
    main()
