import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run_activity_monitoring_app():
    """
    Displays or simulates basic activity monitoring data like heart rate, steps, etc.
    """
    st.title("Activity Monitoring")
    st.markdown("""
    This section demonstrates basic activity monitoring, 
    such as heart rate and step tracking over time.
    """)

    # Let user pick how many days to simulate
    num_days = st.slider("Number of days to simulate:", 7, 60, 14)
    
    # Generate synthetic data (replace with real device data if desired)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_days)
    heart_rates = np.random.randint(60, 120, size=num_days)
    steps = np.random.randint(1000, 10000, size=num_days)
    
    df_activity = pd.DataFrame({
        "Date": dates,
        "Heart_Rate": heart_rates,
        "Steps": steps
    })
    
    # Show data
    st.write("Synthetic Activity Data Preview:")
    st.dataframe(df_activity)
    
    # Plot heart rate over time
    fig_hr = px.line(df_activity, x="Date", y="Heart_Rate", title="Heart Rate Over Time")
    st.plotly_chart(fig_hr, use_container_width=True)
    
    # Plot steps over time
    fig_steps = px.line(df_activity, x="Date", y="Steps", title="Steps Over Time")
    st.plotly_chart(fig_steps, use_container_width=True)
    
    st.write("""
    "Thank you for using our Health Monitoring Dashboard! If you have any questions about your health data, please consult a healthcare professional."
    """)

