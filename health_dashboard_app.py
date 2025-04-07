import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def run_health_dashboard_app():
    """
    A senior-friendly Health Monitoring Dashboard.
    Lets users upload a CSV of health metrics (e.g., glucose, sleep hours)
    and explore them in a simpler, larger-text interface.
    """

    # Larger, more readable title
    st.markdown(
        "<h1 style='text-align: center; font-size: 2em; color: #2F4F4F;'>"
        "Health Monitoring Dashboard (Senior-Friendly)</h1>",
        unsafe_allow_html=True
    )

    # Gentle instructions with larger text
    st.markdown(
        "<p style='font-size: 1.2em;'>"
        "Welcome! This dashboard lets you upload a CSV file containing your health data—such as "
        "glucose levels or hours of sleep. Once uploaded, you'll see a simple preview of your data, "
        "some basic statistics, and a chart to help visualize your information.<br><br>"
        "If you have any difficulty reading the text, you can increase your browser's zoom level "
        "or ask for assistance."
        "</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        label="Upload Your Health CSV Here (e.g., glucose, sleep hours)",
        type=["csv"]
    )

    # Check if user uploaded a file
    if uploaded_file:
        df_health = pd.read_csv(uploaded_file)

        # Data preview section
        st.markdown(
            "<h2 style='font-size: 1.6em; margin-top: 1em;'>Data Preview</h2>",
            unsafe_allow_html=True
        )
        st.dataframe(df_health.head(10), height=300)

        # Descriptive Statistics
        st.markdown(
            "<h3 style='font-size: 1.4em; margin-top: 1em;'>Descriptive Statistics</h3>",
            unsafe_allow_html=True
        )
        st.write(df_health.describe(include='all'))

        # Numeric columns for histogram
        numeric_cols = df_health.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown(
                "<h3 style='font-size: 1.4em; margin-top: 1em;'>Chart Viewer</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='font-size: 1.1em;'>Select a numeric column to see its distribution:</p>",
                unsafe_allow_html=True
            )

            selected_col = st.selectbox(
                "Choose a column to plot",
                numeric_cols
            )

            fig_hist = px.histogram(
                df_health,
                x=selected_col,
                nbins=20,
                title=f"Distribution of {selected_col}",
                template="simple_white"
            )
            fig_hist.update_layout(
                title_font_size=18,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                font=dict(size=14)  # General chart font size
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        else:
            st.warning("It looks like there are no numeric columns in your CSV.")

        # Gentle concluding message
        st.markdown(
            "<p style='font-size: 1.2em; margin-top: 1em;'>"
            "Thank you for using our Health Monitoring Dashboard! If you have any questions "
            "about your health data, please consult a healthcare professional."
            "</p>",
            unsafe_allow_html=True
        )

    else:
        # If no file is uploaded, guide the user
        st.markdown(
            "<p style='font-size: 1.2em; color: #808080;'>"
            "Please upload a CSV with your health metrics. Examples might include daily glucose, "
            "sleep hours, or step counts. Once uploaded, you'll see a preview and chart below."
            "</p>",
            unsafe_allow_html=True
        )
