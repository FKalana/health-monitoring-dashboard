import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import io

# --- Sample Data ---
def load_sample_data():
    return pd.DataFrame({
        "Date": pd.date_range(start="2023-10-01", periods=10, freq='D'),
        "Heart_Rate": [85, 90, 72, 95, 88, 78, 82, 93, 70, 76],
        "Steps": [4500, 6200, 3000, 8000, 5200, 4000, 6500, 7800, 3100, 5000],
        "Glucose": [110, 130, 105, 140, 125, 100, 115, 140, 95, 108],
        "Sleep_Hours": [6.5, 7.0, 6.0, 5.5, 7.2, 8.0, 6.8, 7.4, 5.5, 6.0],
        "Epilepsy_Events": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "Respiration_Rate": [16, 18, 17, 19, 16, 15, 18, 20, 17, 16],
        "Mental_Health_Score": [7, 5, 6, 4, 5, 7, 6, 5, 4, 6]
    })

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def get_summary_stats(df):
    return df.describe(include='all')

# --- PDF Report Generator (Fixed for Streamlit) ---
def generate_pdf_report(data):
    buffer = io.BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    name = data["Name"].iloc[0]
    date = data["Date"].iloc[0]
    glucose = data["Glucose"].iloc[0]
    heart_rate = data["Heart_Rate"].iloc[0]
    sleep = data["Sleep_Hours"].iloc[0]
    respiration = data["Respiration_Rate"].iloc[0]
    mental = data["Mental_Health_Score"].iloc[0]
    epilepsy = "Yes" if data["Epilepsy_Events"].iloc[0] == 1 else "No"

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"{name}'s Health Report - {date}", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Name: {name}", ln=True)
    pdf.cell(0, 10, f"Date: {date}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Health Metrics & Analysis:", ln=True)
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, f"Glucose: {glucose} mg/dL")
    pdf.multi_cell(0, 10, "Warning: High glucose level." if glucose > 125 else "Glucose level is normal.")
    pdf.multi_cell(0, 10, f"Heart Rate: {heart_rate} bpm")
    pdf.multi_cell(0, 10, f"Sleep Hours: {sleep} hrs")
    pdf.multi_cell(0, 10, "Low sleep duration." if sleep < 6 else "Sleep duration is good.")
    pdf.multi_cell(0, 10, f"Respiration Rate: {respiration} breaths/min")
    pdf.multi_cell(0, 10, "Abnormal respiration." if respiration < 12 or respiration > 20 else "Respiration rate normal.")
    pdf.multi_cell(0, 10, f"Mental Health Score: {mental}/10")
    pdf.multi_cell(0, 10, "Consider support." if mental < 5 else "Mental health score is OK.")
    pdf.multi_cell(0, 10, f"Epileptic Event Today: {epilepsy}")
    pdf.ln(5)
    pdf.set_font("Arial", "I", 11)
    pdf.multi_cell(0, 10, "This report is for personal tracking. Consult a doctor for diagnosis.")

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer

# --- Streamlit UI Setup ---
st.set_page_config(layout="centered", page_title="Health Dashboard")
font_size = st.sidebar.selectbox("Font Size", ["Small", "Medium", "Large"], index=1)
font_map = {"Small": "13px", "Medium": "16px", "Large": "19px"}
font_css = font_map[font_size]
st.markdown(f"""<style>html, body, [class*='css']{{font-size: {font_css} !important;}}</style>""", unsafe_allow_html=True)

# --- Load Data ---
if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True
if st.session_state.show_welcome:
    st.markdown("<h2 style='text-align: center;'>Welcome to the Health Monitoring Dashboard</h2>", unsafe_allow_html=True)
    if st.button("Enter Dashboard"):
        st.session_state.show_welcome = False
        st.rerun()
    st.stop()

st.markdown("""
    <h1 style='text-align: center; font-size: 2.2em;'>Your Personal Health Dashboard</h1>
    <p style='text-align: center;'>Track your health and get personalized analysis.</p>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
useload = st.checkbox("Or load demo data")
if uploaded_file:
    df = load_data(uploaded_file)
elif useload:
    df = load_sample_data()
else:
    st.warning("Upload a file or load sample data to continue.")
    st.stop()

if df.isnull().sum().sum() > 0:
    df = df.dropna()

if 'Class' not in df.columns:
    df['Class'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
model = RandomForestClassifier().fit(df[['Glucose', 'Heart_Rate', 'Sleep_Hours']], df['Class'])

# --- Tabs ---
tabs = st.tabs(["Overview", "Visualize", "Insights", "Predict", "Real-Time Tracker", "Export"])

with tabs[0]:
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.subheader("Summary Statistics")
    st.write(get_summary_stats(df))

with tabs[1]:
    st.subheader("Charts")
    col_to_plot = st.multiselect("Select columns to plot:", df.select_dtypes(include=np.number).columns.tolist(), default=['Heart_Rate', 'Glucose'])
    for col in col_to_plot:
        st.plotly_chart(px.line(df, x="Date", y=col, title=f"{col} Over Time"), use_container_width=True)

with tabs[2]:
    st.subheader("Health Recommendations")
    if df['Glucose'].mean() > 125:
        st.error("High average glucose level.")
    if df['Sleep_Hours'].mean() < 6:
        st.warning("Low average sleep duration.")
    if df['Epilepsy_Events'].sum() > 0:
        st.warning("Epileptic events detected.")
    if df['Respiration_Rate'].mean() > 20 or df['Respiration_Rate'].mean() < 12:
        st.warning("Abnormal respiration rate.")
    if df['Mental_Health_Score'].mean() < 5:
        st.warning("Low mental health score.")
    st.success("You're doing great! Keep tracking consistently.")

with tabs[3]:
    st.subheader("Diabetes Risk Prediction")
    glucose = st.slider("Glucose", 70, 200, 120)
    hr = st.slider("Heart Rate", 50, 120, 80)
    sleep = st.slider("Sleep Hours", 3.0, 10.0, 6.0)
    prediction = model.predict([[glucose, hr, sleep]])[0]
    st.write(f"Prediction: {'Positive' if prediction else 'Negative'} Diabetes Risk")

with tabs[4]:
    st.subheader("Enter Your Health Data")
    name = st.text_input("Your Name")
    date = st.date_input("Date", pd.Timestamp.now())
    g = st.number_input("Glucose", 50, 300, 110)
    h = st.number_input("Heart Rate", 40, 180, 80)
    s = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    r = st.number_input("Respiration Rate", 10, 30, 16)
    m = st.slider("Mental Health Score", 1, 10, 6)
    e = st.selectbox("Epileptic Event Today?", ["No", "Yes"])

    if st.button("Generate My Report"):
        if not name.strip():
            st.warning("Please enter your name.")
            st.stop()
        report_df = pd.DataFrame({
            "Name": [name], "Date": [date], "Glucose": [g], "Heart_Rate": [h], "Sleep_Hours": [s],
            "Respiration_Rate": [r], "Mental_Health_Score": [m], "Epilepsy_Events": [1 if e == "Yes" else 0]
        })
        st.session_state.personal_report = report_df
        st.success("Personal report generated.")
        st.dataframe(report_df)
        pred = model.predict([[g, h, s]])[0]
        st.write(f"Diabetes Risk Prediction: {'Positive' if pred else 'Negative'}")

    if 'personal_report' in st.session_state:
        csv = st.session_state.personal_report.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download {name}'s Report (CSV)", csv, "my_health_report.csv", "text/csv")
        pdf = generate_pdf_report(st.session_state.personal_report)
        st.download_button(f"Download {name}'s Report (PDF)", data=pdf, file_name="my_health_report.pdf", mime="application/pdf")

with tabs[5]:
    st.subheader("Export Full Dataset")
    full_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Dataset (CSV)", full_csv, "health_data.csv", "text/csv")
