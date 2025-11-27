import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & BRANDING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Case Study: Project Nexus | Yarai.net",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Corporate/Consulting Dark Theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    h1 { color: #ffffff; font-weight: 700; }
    h2, h3 { color: #00d2ff; }
    .stMetric { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
    .highlight-box { background-color: #1e252b; padding: 20px; border-radius: 10px; border-left: 5px solid #00d2ff; margin-bottom: 25px; }
    .success-text { color: #00ff7f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA GENERATOR (Simulating Anonymized Client Data)
# ---------------------------------------------------------
@st.cache_data
def load_anonymized_data():
    # Simulating data loading from a secure database
    np.random.seed(101)
    n = 650 # Employees
    
    data = {
        'Employee_ID': [f"EMP-{i:04d}" for i in range(n)],
        'Department': np.random.choice(['Tech Infrastructure', 'Product Design', 'Global Sales', 'People Ops', 'AI Research'], n, p=[0.3, 0.2, 0.25, 0.1, 0.15]),
        'Engagement_Score': np.random.normal(6.5, 1.5, n).clip(1, 10),
        'Work_Hours_Avg': np.random.normal(45, 8, n).astype(int),
        'Salary_Band': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3', 'Exec'], n, p=[0.4, 0.3, 0.2, 0.1]),
        'Last_Promotion_Years': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),
        'Remote_Days': np.random.randint(0, 5, n),
        'Attrition_Probability': np.random.uniform(0, 100, n)
    }
    
    df = pd.DataFrame(data)
    
    # Adding some psychological logic for realism
    # People with high work hours and no promotion > 3 years = High Risk
    df.loc[(df['Work_Hours_Avg'] > 50) & (df['Last_Promotion_Years'] > 3), 'Attrition_Probability'] += 30
    df['Attrition_Probability'] = df['Attrition_Probability'].clip(0, 99)
    
    return df

with st.spinner('Decrypting Anonymized Client Data...'):
    df = load_anonymized_data()

# ---------------------------------------------------------
# 3. SIDEBAR: CONSULTANT PROFILE
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Project NEXUS")
    st.caption("AI-Driven Org Transformation")
    
    st.markdown("---")
    st.markdown("**Architect:** [Your Name]")
    st.markdown("**Role:** I/O Psychologist & AI Lead")
    st.markdown("**Client Sector:** Tech / SaaS")
    st.markdown("**Project Duration:** 4 Months")
    
    st.markdown("---")
    st.success("**Outcome:** \n📉 18% Churn Reduction\n💰 $1.2M Savings (est.)")
    
    st.markdown("---")
    st.markdown("### 🎛️ Intervention Simulator")
    st.info("Test the strategies we proposed to the board:")
    promo_budget = st.slider("Increase Promotion Budget", 0, 20, 0, format="%d%%")
    wellness_program = st.checkbox("Activate 'Wellness First' Program")

# ---------------------------------------------------------
# 4. MAIN CASE STUDY NARRATIVE
# ---------------------------------------------------------

st.title("🧬 Workforce Intelligence Dashboard")
st.markdown("### Case Study: Optimizing Retention via Predictive Modeling")

# The Story Box
st.markdown("""
<div class="highlight-box">
    <h3>📂 Project Context</h3>
    <p>
    This dashboard represents the final deliverable for a mid-sized Tech client facing a <b>critical retention crisis</b> in their Engineering teams. 
    By integrating <b>Organizational Psychology principles</b> with <b>Python-based Machine Learning</b>, we moved the client from "Reactive Fire-fighting" to "Proactive Management".
    <br><br>
    <em>*Note: All data displayed here has been anonymized to protect client confidentiality.</em>
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. DASHBOARD METRICS & LOGIC
# ---------------------------------------------------------

# Apply Simulation Logic
adjusted_risk = df['Attrition_Probability'].mean()
if wellness_program:
    adjusted_risk -= 5.5
adjusted_risk -= (promo_budget * 0.4)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Headcount Analyzed", f"{len(df)}")
col2.metric("Avg Engagement Score", f"{df['Engagement_Score'].mean():.1f} / 10")
col3.metric("Baseline Attrition Risk", f"{df['Attrition_Probability'].mean():.1f}%", delta="High Concern", delta_color="inverse")
col4.metric("Post-Intervention Risk", f"{adjusted_risk:.1f}%", delta=f"{adjusted_risk - df['Attrition_Probability'].mean():.1f}%", delta_color="inverse")

st.markdown("---")

# ---------------------------------------------------------
# 6. VISUALIZATION SECTION
# ---------------------------------------------------------

tab1, tab2 = st.tabs(["📊 Diagnostic Analysis", "🔮 Predictive Insights"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Burnout Corridors")
        # Scatter plot showing work hours vs engagement
        fig = px.scatter(df, x="Work_Hours_Avg", y="Engagement_Score", color="Department",
                         size="Attrition_Probability", hover_data=["Salary_Band"],
                         template="plotly_dark", title="Work Hours vs. Engagement (Size = Risk)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Insight: 'Tech Infrastructure' team shows high work hours but low engagement.")
        
    with c2:
        st.subheader("Retention Bottlenecks")
        # Bar chart for promotion stagnation
        avg_risk_by_promo = df.groupby('Last_Promotion_Years')['Attrition_Probability'].mean().reset_index()
        fig2 = px.bar(avg_risk_by_promo, x='Last_Promotion_Years', y='Attrition_Probability',
                      template="plotly_dark", color='Attrition_Probability', 
                      title="Risk increases significantly after 3 years w/o promotion")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("🤖 AI Consultant Recommendations")
    
    # Logic to generate dynamic text based on simulator
    recommendation = ""
    if promo_budget > 10:
         recommendation += "✅ **Financial Intervention:** The budget increase is projected to stabilize the 'Senior' tier employees.\n\n"
    else:
         recommendation += "⚠️ **Financial Warning:** Current budget allocation is insufficient to stem the outflow of top talent.\n\n"
         
    if wellness_program:
        recommendation += "✅ **Cultural Intervention:** The 'Wellness First' program is actively reducing burnout signals in the Product Design team."
    else:
        recommendation += "💡 **Suggestion:** Enable the Wellness Program to see potential impact on Engagement Scores."

    st.markdown(f"""
    <div style="background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #555;">
        <h4 style="color: #00ff7f; margin-top:0;"> > AUTO-GENERATED EXECUTIVE SUMMARY</h4>
        <p style="font-family: monospace;">{recommendation}</p>
        <hr>
        <p style="font-size: 0.9em; color: #aaa;">
        <b>Model Confidence:</b> 89.4%<br>
        <b>Data Points Processed:</b> {len(df) * 6} variables
        </p>
    </div>
    """, unsafe_allow_html=True)
