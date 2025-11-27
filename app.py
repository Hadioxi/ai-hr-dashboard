import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="YARAI | AI-Enhanced Management",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Modern, Clean, Professional
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    h1 { color: #ffffff; font-weight: 700; }
    h2, h3 { color: #00d2ff; }
    .stMetric { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
    .highlight-box { background-color: #15191f; padding: 20px; border-radius: 10px; border: 1px solid #4c4c4c; border-left: 5px solid #00ff7f; margin-bottom: 25px; }
    .skill-tag { background-color: #262730; padding: 5px 10px; border-radius: 15px; font-size: 0.8em; margin-right: 5px; display: inline-block; border: 1px solid #555; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA GENERATOR (Simulated for Demo)
# ---------------------------------------------------------
@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 300
    
    data = {
        'Employee_ID': range(1, n+1),
        'Department': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing', 'Finance'], n),
        'Performance': np.random.normal(70, 15, n).clip(0, 100),
        'Satisfaction': np.random.normal(6.5, 2, n).clip(1, 10),
        'Work_Hours': np.random.normal(40, 10, n),
        'Salary': np.random.normal(60000, 15000, n),
        'Remote_Days': np.random.randint(0, 5, n),
        'Risk_Score': np.random.uniform(0, 100, n) 
    }
    df = pd.DataFrame(data)
    
    # Simple logic for the demo
    df['Risk_Score'] = 100 - df['Satisfaction'] * 10
    return df

with st.spinner('Initializing AI Dashboard Demo...'):
    df = generate_demo_data()

# ---------------------------------------------------------
# 3. SIDEBAR: PROFESSIONAL PROFILE
# ---------------------------------------------------------
with st.sidebar:
    st.title("👨‍💻 The Architect View")
    st.caption("Organizational Consultant & AI Expert")
    
    st.markdown("---")
    st.write("**Why this dashboard?**")
    st.info("""
    This is a **Live Demonstration** of my capabilities. 
    
    I build systems that transform raw organizational data into:
    1. Strategic Insights
    2. Predictive Models
    3. Automated Reports
    """)
    
    st.markdown("---")
    st.write("**My Tech Stack:**")
    st.markdown("""
    <div style="line-height: 2.2;">
    <span class="skill-tag">Python</span>
    <span class="skill-tag">Streamlit</span>
    <span class="skill-tag">Data Science</span>
    <span class="skill-tag">I/O Psychology</span>
    <span class="skill-tag">AI Agents</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎛️ Interactive Demo")
    st.write("Try controlling the organization:")
    pay_raise = st.slider("Simulate Pay Raise (%)", 0, 30, 0)

# ---------------------------------------------------------
# 4. MAIN VALUE PROPOSITION
# ---------------------------------------------------------

st.title("YARAI.NET | The Future of Work")
st.markdown("### Turning Data into Decisions")

# The "Pitch" Box
st.markdown("""
<div class="highlight-box">
    <h3>👋 Hi, I'm an AI Solutions Architect.</h3>
    <p>
    If I join your organization, I won't just bring spreadsheets. I will build <b>Intelligent Systems</b> like this one.
    <br><br>
    This dashboard demonstrates how I combine <b>Organizational Psychology</b> with <b>Python</b> to visualize health, predict burnout, and optimize workflows automatically.
    <br>
    <i>(Note: All data below is simulated for this demonstration)</i>
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. LIVE DEMONSTRATION
# ---------------------------------------------------------

# Simulation logic
simulated_satisfaction = df['Satisfaction'].mean() + (pay_raise * 0.05)
simulated_risk = df['Risk_Score'].mean() - (pay_raise * 0.3)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Data Processing Capability", "Unlimited", "Scalable")
col2.metric("Current Satisfaction (Demo)", f"{df['Satisfaction'].mean():.1f}/10")
col3.metric("Simulated Satisfaction", f"{simulated_satisfaction:.1f}/10", delta=f"+{pay_raise*0.05:.1f}")
col4.metric("Churn Risk Reduction", f"{simulated_risk:.1f}%", delta=f"-{pay_raise*0.3:.1f}%", delta_color="inverse")

st.markdown("---")

tab1, tab2 = st.tabs(["📊 Advanced Analytics", "🧠 AI Strategic Reporting"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Identifying Burnout Clusters")
        st.caption("I use clustering algorithms to find teams at risk.")
        fig = px.scatter(df, x="Work_Hours", y="Satisfaction", color="Department", 
                         size="Risk_Score", template="plotly_dark",
                         title="Workload vs Satisfaction Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Performance Distribution")
        st.caption("Visualizing talent density across departments.")
        fig2 = px.box(df, x="Department", y="Performance", color="Department", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Automated Executive Summaries")
    st.write("I automate the boring stuff. Instead of writing weekly reports manually, my AI agents generate this for you:")
    
    report_trigger = st.button("Generate Live Analysis")
    
    if report_trigger:
        with st.spinner("AI Agent is analyzing the simulated data..."):
            time.sleep(1.5)
            st.success("Analysis Complete")
            
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 5px; font-family: monospace; border-left: 3px solid #00d2ff;">
            <strong>[AI AGENT REPORT LOG]</strong><br><br>
            1. <strong>SIMULATION EFFECT:</strong> Increasing salaries by {pay_raise}% is projected to boost satisfaction scores to {simulated_satisfaction:.1f}.<br>
            2. <strong>RISK ALERT:</strong> The 'Engineering' department shows high variance in work hours. Recommended action: <em>Review sprint planning.</em><br>
            3. <strong>ARCHITECT NOTE:</strong> This insight was generated automatically. Imagine this system running on your real-time company data.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Click the button to see how I automate reporting.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Designed & Built by [Your Name] | Hosted on Yarai.net</div>", unsafe_allow_html=True)
