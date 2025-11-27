import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# ---------------------------------------------------------
# 1. SETUP & CYBER-THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="YARAI | AI Workforce Architect",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" # باز شدن تمام صفحه برای تاثیر بیشتر
)

# CSS for Matrix/Cyberpunk Vibe
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    h1 { color: #00f2ea; text-shadow: 0 0 10px #00f2ea; font-family: 'Courier New'; }
    h2, h3 { color: #bd00ff; }
    .metric-card { background: linear-gradient(145deg, #111, #161616); border: 1px solid #333; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,242,234,0.1); }
    .highlight { color: #00f2ea; font-weight: bold; }
    .terminal { background-color: #000; color: #33ff00; font-family: 'Courier New'; padding: 15px; border-radius: 5px; border: 1px solid #33ff00; height: 150px; overflow-y: scroll; font-size: 0.85em; }
    div[data-testid="stExpander"] details summary { color: #00f2ea; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. ADVANCED DATA SIMULATION
# ---------------------------------------------------------
@st.cache_data
def load_complex_data():
    n = 200
    departments = ['Engineering', 'Sales', 'Product', 'HR', 'Executive']
    
    df = pd.DataFrame({
        'ID': range(n),
        'Name': [f"Node_{i}" for i in range(n)],
        'Dept': np.random.choice(departments, n),
        'Sentiment': np.random.uniform(-1, 1, n), # -1 bad, +1 good
        'Performance': np.random.normal(75, 15, n).clip(0, 100),
        'Burnout_Risk': np.random.uniform(0, 100, n),
        'Influence_Score': np.random.randint(10, 100, n) # Network Centrality
    })
    
    # Correlation: High Burnout -> Low Sentiment
    df['Sentiment'] = df['Sentiment'] - (df['Burnout_Risk'] / 200)
    return df

df = load_complex_data()

# ---------------------------------------------------------
# 3. SIDEBAR (YOUR PROFILE)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11184/11184128.png", width=100)
    st.markdown("## 👨‍💻 ARCHITECT PROFILE")
    st.info("""
    **Name:** [Your Name]
    **Role:** Org Consultant & AI Architect
    **Mission:** Automating Org Intelligence.
    """)
    
    st.markdown("### 🛠️ Capabilities")
    st.progress(95, text="Python & AI Agents")
    st.progress(90, text="Organizational Psychology")
    st.progress(85, text="Data Visualization")
    
    st.write("---")
    st.write("📧 Contact: hi@yarai.net")

# ---------------------------------------------------------
# 4. HEADER & HERO SECTION
# ---------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("YARAI // ORG.OS_v4.0")
    st.markdown("#### The Operating System for Modern Organizations")
    st.markdown("> *You bring the data. I bring the intelligence.*")

with c2:
    if st.button("⚡ ACTIVATE DEMO MODE"):
        st.toast("System Online. AI Agents Deployed.", icon="🤖")

st.write("---")

# ---------------------------------------------------------
# 5. MAIN DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🌐 Network Intelligence (ONA)", "🧠 NLP & Sentiment", "🔮 Predictive Simulator"])

# --- TAB 1: THE COOL NETWORK GRAPH ---
with tab1:
    st.markdown("### 🕸️ Organizational Network Analysis")
    st.markdown("Most consultants look at Org Charts. I look at **Real Interactions**. This graph visualizes communication silos.")
    
    # Simulating a Network Graph with Plotly
    # Generate random connections
    edge_x = []
    edge_y = []
    node_x = np.random.randn(200)
    node_y = np.random.randn(200)
    
    for i in range(len(node_x)):
        # Connect to 2 random other nodes
        target = np.random.randint(0, 200)
        edge_x.extend([node_x[i], node_x[target], None])
        edge_y.extend([node_y[i], node_y[target], None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#444'), hoverinfo='none', mode='lines')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='Viridis', size=10, color=df['Influence_Score'], line_width=2),
        text=df['Dept'] + ": " + df['Performance'].astype(str)
    )

    fig_net = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False, hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
    st.plotly_chart(fig_net, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Isolated Nodes Detected", "14", delta="Risk", delta_color="inverse")
    c2.metric("Network Density", "0.45", "Optimal")
    c3.metric("Info Bottlenecks", "3 Key Players", "Critical")

# --- TAB 2: NLP & SENTIMENT ---
with tab2:
    st.subheader("💬 AI Sentiment Decoding")
    st.write("I use **Natural Language Processing (NLP)** to understand employee morale from slack messages/surveys (Simulated).")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_scatter = px.scatter(df, x="Performance", y="Sentiment", color="Dept", size="Influence_Score",
                                 template="plotly_dark", title="Performance vs. Sentiment Correlation")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown("**Real-time Topic Modeling:**")
        st.warning("⚠️ Topic: 'Burnout' (Trending up 15%)")
        st.success("✅ Topic: 'New Product' (Positive Sentiment)")
        st.info("ℹ️ Topic: 'Remote Policy' (Mixed Sentiment)")
        
        st.markdown("---")
        st.write("*My agents automatically flag toxic environments before people quit.*")

# --- TAB 3: SIMULATOR ---
with tab3:
    st.subheader("🎛️ Strategic Simulator")
    st.write("Don't guess. Simulate.")
    
    col_input, col_output = st.columns(2)
    with col_input:
        salary = st.slider("Budget Increase ($)", 0, 1000000, 200000)
        wfh = st.slider("Remote Days / Week", 0, 5, 2)
        training = st.checkbox("Implement AI Training Program?", value=True)
    
    with col_output:
        # Fake Math Logic
        retention_gain = (salary / 50000) + (wfh * 1.5) + (5 if training else 0)
        cost_saving = retention_gain * 12000
        
        st.metric("Projected Retention Gain", f"+{retention_gain:.1f}%")
        st.metric("Est. Recruitment Savings", f"${cost_saving:,.0f}")
        
        if retention_gain > 15:
             st.success("✅ Strategy Approved by AI Model")
        else:
             st.error("❌ Impact too low. Adjust parameters.")

# ---------------------------------------------------------
# 6. THE TERMINAL (Wow Factor)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📟 YARAI System Logs (Live)")

terminal_placeholder = st.empty()
log_lines = [
    "[INFO] Initializing Neural Modules...",
    "[INFO] Connected to Employee_DB (Encrypted)",
    "[AGENT_01] Scanning for 'Quiet Quitting' patterns...",
    "[AGENT_02] Analyzing 4,500 Slack messages for sentiment...",
    "[WARN] Detected anomaly in Engineering Dept (Burnout Risk > 85%)",
    "[AUTO] Generating executive summary...",
    "[SUCCESS] Dashboard updated successfully.",
    "[IDLE] Awaiting user input..."
]

# Simple animation simulation
terminal_text = ""
for line in log_lines:
    terminal_text += f"<span style='color:#00f2ea'>{datetime.now().strftime('%H:%M:%S')}</span> {line}<br>"
    
terminal_placeholder.markdown(f"<div class='terminal'>{terminal_text}</div>", unsafe_allow_html=True)
