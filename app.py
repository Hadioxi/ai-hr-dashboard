import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="YARAI | Enterprise AI Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark/Tech" Vibe
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #1e252b; padding: 10px; border-radius: 8px; border-left: 5px solid #00d2ff; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    .report-box { background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #4c4c4c; font-family: monospace; color: #00ff7f;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. FAKE DATA GENERATOR (Advanced)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Loading animation simulation
    time.sleep(1.5) 
    
    np.random.seed(42)
    n_employees = 500
    departments = ['R&D', 'Engineering', 'Sales', 'HR', 'Marketing', 'Data Science']
    
    data = {
        'ID': range(1001, 1001 + n_employees),
        'Department': np.random.choice(departments, n_employees, p=[0.1, 0.3, 0.25, 0.1, 0.15, 0.1]),
        'Age': np.random.randint(22, 58, n_employees),
        'Tenure_Years': np.random.randint(1, 15, n_employees),
        'Salary': np.random.normal(70000, 20000, n_employees).astype(int),
        'Performance_Score': np.random.normal(75, 15, n_employees).clip(0, 100),
        'Burnout_Index': np.random.uniform(0, 10, n_employees), # 10 = High Burnout
        'Collaboration_Score': np.random.uniform(0, 100, n_employees), # Network centrality
        'Remote_Days': np.random.randint(0, 5, n_employees)
    }
    
    df = pd.DataFrame(data)
    
    # Logic: High Burnout + Low Pay = High Risk
    df['Attrition_Risk_Prob'] = (
        (df['Burnout_Index'] / 10) * 0.4 + 
        (1 - (df['Salary'] / df['Salary'].max())) * 0.4 + 
        (np.random.rand(n_employees) * 0.2)
    )
    
    return df

# ---------------------------------------------------------
# 3. SIDEBAR & CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧠 YARAI.NET")
    st.caption("AI-Powered Organizational Architect")
    st.markdown("---")
    
    st.subheader("⚙️ Control Panel")
    selected_dept = st.multiselect("Filter Departments", ['R&D', 'Engineering', 'Sales', 'HR', 'Marketing', 'Data Science'], default=['Engineering', 'Data Science', 'Sales'])
    
    st.markdown("### 🤖 AI Simulator")
    st.info("Adjust parameters to see predicted impact on Retention.")
    salary_bump = st.slider("Simulate Salary Increase (%)", 0, 30, 0)
    remote_policy = st.slider("Remote Work Days / Week", 0, 5, 2)
    
    st.markdown("---")
    st.write("© 2025 Yarai.net | Internal Build v4.2")

# Load Data
with st.spinner('Connecting to Neural Database...'):
    raw_df = load_data()

# Filter Data
if selected_dept:
    df = raw_df[raw_df['Department'].isin(selected_dept)].copy()
else:
    df = raw_df.copy()

# Apply Simulation Logic (Simple Math for Demo)
df['Simulated_Risk'] = df['Attrition_Risk_Prob'] - (salary_bump/100 * 0.5) - (remote_policy * 0.02)
df['Simulated_Risk'] = df['Simulated_Risk'].clip(0, 1)

# ---------------------------------------------------------
# 4. MAIN DASHBOARD UI
# ---------------------------------------------------------

st.title("Enterprise Workforce Intelligence")
st.markdown("Real-time analysis of **Human Capital**, **Psychological Safety**, and **AI-Driven Retention Models**.")

# TOP METRICS ROW
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
avg_risk = df['Attrition_Risk_Prob'].mean() * 100
sim_risk = df['Simulated_Risk'].mean() * 100

kpi1.metric("Active Workforce", f"{len(df):,}", "+12 this month")
kpi2.metric("Avg Burnout Index", f"{df['Burnout_Index'].mean():.1f} / 10", delta="-0.2 (Improving)")
kpi3.metric("Current Attrition Risk", f"{avg_risk:.1f}%", delta_color="off")
kpi4.metric("Predicted Risk (After Sim)", f"{sim_risk:.1f}%", delta=f"{sim_risk - avg_risk:.1f}%", delta_color="inverse")

st.markdown("---")

# TABS FOR DIFFERENT VIEWS
tab1, tab2, tab3 = st.tabs(["📊 Performance & Burnout", "🕸️ Network Analysis (ONA)", "📝 AI Strategic Report"])

# --- TAB 1: SCATTER & DISTRIBUTION ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Psychological Safety vs. Performance")
        # 3D Scatter looks very "Tech"
        fig_3d = px.scatter_3d(df, x='Burnout_Index', y='Performance_Score', z='Salary',
                               color='Department', size='Collaboration_Score',
                               opacity=0.7, template="plotly_dark",
                               title="Multi-dimensional Employee Analysis")
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with col2:
        st.subheader("Risk Distribution")
        # Histogram showing who is likely to leave
        fig_hist = px.histogram(df, x="Attrition_Risk_Prob", color="Department", 
                                nbins=20, template="plotly_dark",
                                title="Flight Risk Probability Distribution")
        fig_hist.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 2: SIMULATED NETWORK ---
with tab2:
    st.subheader("Organizational Network Analysis (ONA)")
    st.markdown("Visualizing hidden silos and communication bottlenecks using **Graph Theory**.")
    
    # Simulating a bubble chart that looks like clusters
    fig_bubble = px.scatter(df, x="Tenure_Years", y="Collaboration_Score", 
                            size="Salary", color="Department",
                            hover_name="ID", size_max=40,
                            template="plotly_dark", title="Collaboration Centrality vs Tenure")
    
    # Adding lines to simulate network connections (Fake visual)
    fig_bubble.add_shape(type="line", x0=df['Tenure_Years'].mean(), y0=0, x1=df['Tenure_Years'].mean(), y1=100,
                         line=dict(color="white", width=1, dash="dash"))
    
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    c1, c2 = st.columns(2)
    c1.info("💡 **Insight:** 'Data Science' team is becoming isolated from 'Sales'. Cross-functional workshops recommended.")
    c2.warning("⚠️ **Alert:** Key influencers in 'Engineering' show high burnout risk.")

# --- TAB 3: AI GENERATED REPORT ---
with tab3:
    st.subheader("🧠 Automated Consultant Insight")
    
    if st.button("Generate AI Report"):
        report_text = f"""
        [SYSTEM INITIALIZED]
        [ANALYZING DATA POINTS: {len(df) * 8}]
        [RUNNING PREDICTIVE MODELS... DONE]

        EXECUTIVE SUMMARY FOR STAKEHOLDERS:
        
        1. TALENT RETENTION WARNING:
           Based on current telemetry, the {df['Department'].value_counts().idxmax()} department faces a critical period.
           Burnout indices are trending upward by 14% quarter-over-quarter.
           
        2. SIMULATION RESULTS:
           Your adjustment of Salary (+{salary_bump}%) and Remote Work ({remote_policy} days) 
           is projected to SAVE approximately {int(len(df) * (avg_risk - sim_risk)/100)} employees from churn.
           Estimated Cost Saving: ${int(len(df) * (avg_risk - sim_risk)/100) * 15000:,} (Recruitment Costs).

        3. STRATEGIC RECOMMENDATION:
           The link between 'Collaboration Score' and 'Performance' is strong (r=0.65). 
           Invest in collaborative tools rather than strict monitoring.
           
        [END OF REPORT]
        """
        
        # Typewriter effect
        t = st.empty()
        for i in range(len(report_text) + 1):
            t.markdown(f'<div class="report-box">{report_text[:i]}</div>', unsafe_allow_html=True)
            time.sleep(0.005) # Speed of typing
    else:
        st.markdown('<div class="report-box">Press the button to generate analysis...</div>', unsafe_allow_html=True)
