import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page Configuration
st.set_page_config(page_title="Team AI | HR Analytics", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🤖 Team AI Dashboard")
st.sidebar.info("Organizational Consultant & AI Solutions Architect")
department_filter = st.sidebar.multiselect(
    "Select Departments:",
    options=["Engineering", "Sales", "HR", "Marketing", "Finance"],
    default=["Engineering", "Sales", "HR"]
)

# Fake Data Generation
def generate_data():
    data = {
        'Department': np.random.choice(["Engineering", "Sales", "HR", "Marketing", "Finance"], 200),
        'Employee_Satisfaction': np.random.randint(1, 10, 200),
        'Performance_Score': np.random.randint(1, 100, 200),
        'Work_Hours': np.random.randint(30, 60, 200),
        'Attrition_Risk': np.random.choice(['Low', 'Medium', 'High'], 200, p=[0.7, 0.2, 0.1])
    }
    return pd.DataFrame(data)

df = generate_data()

# Filter Data
if department_filter:
    df = df[df['Department'].isin(department_filter)]

# Header
st.title("📊 Organizational Health & AI Analytics")
st.markdown("Real-time insights into workforce dynamics using AI-driven prediction models.")

# KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Employees", len(df))
c2.metric("Avg Satisfaction", f"{df['Employee_Satisfaction'].mean():.1f}/10")
c3.metric("High Performance Rate", f"{len(df[df['Performance_Score']>80])/len(df)*100:.0f}%")
c4.metric("AI Churn Prediction", f"{len(df[df['Attrition_Risk']=='High'])} Staff at Risk", delta="-2", delta_color="inverse")

# Charts Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Departmental Performance Analysis")
    fig_scatter = px.scatter(df, x="Work_Hours", y="Performance_Score", color="Department", 
                             size="Employee_Satisfaction", hover_data=['Attrition_Risk'],
                             template="plotly_white", title="Work Hours vs Performance (AI Clustering)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("Employee Satisfaction Distribution")
    fig_hist = px.histogram(df, x="Employee_Satisfaction", color="Department", barmode="group",
                            template="plotly_white", title="Satisfaction Levels by Dept")
    st.plotly_chart(fig_hist, use_container_width=True)

# AI Insights Section
st.markdown("---")
st.subheader("🧠 AI Consultant Insights (Automated Generation)")
st.success(f"""
Based on the analysis of {len(df)} employee records, our AI model suggests:
1. **Engineering** department shows signs of burnout (High Work Hours vs Lower Satisfaction).
2. **Sales** team has the highest correlation between performance and satisfaction.
3. Recommended Action: Initiate a *Flexible Work Program* for the High-Risk group identified in the dashboard.
""")
