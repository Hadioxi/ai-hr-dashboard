import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. SYSTEM CONFIGURATION & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="IR-HRM Intelligent System | 1403",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Crisis Management" Vibe (Enterprise Dark)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Tahoma', 'Segoe UI', sans-serif; color: #ffffff; }
    .metric-box {
        background-color: #1a1f29;
        border-left: 5px solid #d97706; /* Amber for Warning */
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .safe-box { border-left-color: #10b981; } /* Green */
    .danger-box { border-left-color: #ef4444; } /* Red */
    .big-number { font-size: 24px; font-weight: bold; color: #f3f4f6; }
    .small-text { font-size: 12px; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA ENGINE: SIMULATING THE IRANIAN CONTEXT (1402-1403)
# ---------------------------------------------------------
@st.cache_data
def load_strategic_data():
    np.random.seed(1403)
    n = 1000
    
    # Departments
    depts = ['IT & Tech', 'Sales & Marketing', 'Operations', 'Finance', 'R&D']
    
    df = pd.DataFrame({
        'Emp_ID': range(1000, 1000+n),
        'Department': np.random.choice(depts, n, p=[0.25, 0.3, 0.2, 0.1, 0.15]),
        'Tenure_Months': np.random.randint(6, 120, n),
        'Salary_Satisfaction': np.random.normal(4, 2, n).clip(1, 10), # Impact of Inflation
    })
    
    # --- MODELING JD-R (Job Demands-Resources) ---
    # Demands (Chapter 2.1): Role Ambiguity, Techno-Stress, Workload
    df['Role_Ambiguity'] = np.random.normal(5, 2, n).clip(1, 10)
    df['Techno_Stress'] = np.random.normal(4, 2.5, n).clip(1, 10) # High in IT
    df['Total_Demands'] = (df['Role_Ambiguity'] + df['Techno_Stress']) / 2
    
    # Resources: Autonomy, Social Support (The Buffer)
    df['Supervisor_Support'] = np.random.normal(5, 2, n).clip(1, 10)
    df['Autonomy'] = np.random.normal(5, 2, n).clip(1, 10)
    df['Total_Resources'] = (df['Supervisor_Support'] + df['Autonomy']) / 2
    
    # --- PSYCHOLOGICAL CONTRACT (Chapter 2.2) ---
    # Breach: "I worked hard, but inflation killed my purchasing power"
    # Logic: Low Salary Sat + High Tenure = High Feeling of Breach
    df['Contract_Breach_Index'] = (10 - df['Salary_Satisfaction']) * 0.6 + (df['Tenure_Months']/120 * 4)
    df['Contract_Breach_Index'] = df['Contract_Breach_Index'].clip(0, 10)
    
    # --- PREDICTING CHURN (Chapter 4.3 - CatBoost Logic Simulation) ---
    # High Demands + Low Resources + High Breach = High Churn Risk
    risk_score = (
        (df['Total_Demands'] * 0.3) - 
        (df['Total_Resources'] * 0.3) + 
        (df['Contract_Breach_Index'] * 0.4)
    )
    # Normalize Risk to 0-100%
    df['Churn_Prob'] = ((risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())) * 100
    
    # Migration Intent (Specific to 1403)
    # High skill (Tech/R&D) + High Breach = Migration Risk
    df['Migration_Risk'] = np.where(
        (df['Department'].isin(['IT & Tech', 'R&D'])) & (df['Churn_Prob'] > 60), 
        True, False
    )
    
    return df

df = load_strategic_data()

# ---------------------------------------------------------
# 3. SIDEBAR: STRATEGIC CONTEXT
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2312/2312479.png", width=80)
    st.title("سامانه هوشمند نگهداشت")
    st.caption("نسخه سازمانی ۱۴۰۳ | مبتنی بر مدل JD-R")
    
    st.markdown("---")
    st.markdown("### ⚙️ تنظیمات داشبورد")
    risk_threshold = st.slider("آستانه ریسک بحرانی (%)", 50, 90, 70, help="کارکنانی که احتمال ترک خدمت آنها بالاتر از این عدد است.")
    inflation_rate = st.number_input("نرخ تورم انتظاری (تعدیل مدل)", value=40, step=5)
    
    st.info(f"""
    **وضعیت سیستم:** فعال ✅
    **مدل پیش‌بینی:** CatBoost Ensembles
    **تعداد پرسنل پایش شده:** {len(df)}
    """)
    
    st.markdown("---")
    st.write("**طراحی شده بر اساس:** گزارش جامع راهبردی ۱۴۰۲-۱۴۰۳")

# ---------------------------------------------------------
# 4. MAIN DASHBOARD STRUCTURE
# ---------------------------------------------------------

# Header
st.title("کالبدشکافی سرمایه انسانی و پیش‌بینی ترک خدمت")
st.markdown("رصد لحظه‌ای **سلامت سازمانی**، **شکاف قرارداد روانشناختی** و **هزینه خروج نخبگان**.")

# --- SECTION 1: MACRO VIEW (CEO DASHBOARD) ---
st.markdown("### 📊 وضعیت کلان سازمان (CEO View)")

# Calculating Metrics
high_risk_staff = df[df['Churn_Prob'] > risk_threshold]
migration_candidates = df[df['Migration_Risk'] == True]
# Cost calculation: Assuming replacement cost = 300M Tomans (Recruitment + Onboarding + Lost Productivity)
turnover_cost = len(high_risk_staff) * 300000000 / 1000000000 # In Billions

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box danger-box">
        <div class="small-text">ریسک "بحران خاموش"</div>
        <div class="big-number">{len(high_risk_staff)} نفر</div>
        <div class="small-text">احتمال خروج > {risk_threshold}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box safe-box">
        <div class="small-text">شاخص سلامت (JD-R Ratio)</div>
        <div class="big-number">{(df['Total_Resources'].mean() / df['Total_Demands'].mean()):.2f}</div>
        <div class="small-text">هدف: > 1.0 (توازن منابع/الزامات)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box metric-box">
        <div class="small-text">هزینه فرصت از دست رفته</div>
        <div class="big-number">{turnover_cost:.1f} Mld T</div>
        <div class="small-text">میلیارد تومان (برآورد جایگزینی)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box danger-box">
        <div class="small-text">سیگنال مهاجرت (Elite Flight)</div>
        <div class="big-number">{len(migration_candidates)}</div>
        <div class="small-text">نخبگان Tech و R&D در خطر</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. DEEP DIVE TABS
# ---------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🧠 تحلیل قرارداد روانشناختی & JD-R", "🌪️ نقشه حرارتی ریسک", "💊 راهکارهای مداخله (Action)"])

# --- TAB 1: THE PSYCHOLOGY (JD-R Model) ---
with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("تحلیل مدل الزامات-منابع (JD-R)")
        st.caption("آیا 'منابع شغلی' (حمایت، استقلال) توانسته‌اند فشار 'الزامات' (ابهام نقش، تورم) را خنثی کنند؟")
        
        fig_scatter = px.scatter(
            df, x="Total_Demands", y="Total_Resources", color="Churn_Prob",
            size="Contract_Breach_Index", hover_data=['Department'],
            color_continuous_scale="RdYlGn_r", # Red = High Churn
            labels={"Total_Demands": "فشارها و الزامات شغلی", "Total_Resources": "منابع و حمایت سازمانی"},
            title="ترازوی فرسودگی: ناحیه پایین-راست (فشار بالا/حمایت کم) = ناحیه خطر",
            template="plotly_dark", height=500
        )
        # Adding quadrants
        fig_scatter.add_hline(y=5, line_dash="dash", line_color="white")
        fig_scatter.add_vline(x=5, line_dash="dash", line_color="white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with c2:
        st.subheader("شاخص نقض قرارداد")
        st.markdown("""
        > **تئوری:** وقتی کارکنان احساس کنند "تورم" تلاش‌هایشان را بی‌اثر کرده، دچار **استعفای خاموش** می‌شوند.
        """)
        
        breach_by_dept = df.groupby('Department')['Contract_Breach_Index'].mean().sort_values(ascending=False)
        fig_bar = px.bar(breach_by_dept, orientation='h', 
                         color=breach_by_dept.values, color_continuous_scale="Reds",
                         title="میانگین احساس 'بی‌عدالتی' به تفکیک واحد",
                         template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: RISK MAP (Operational View) ---
with tab2:
    st.subheader("رصدخانه استراتژیک: کانون‌های بحران کجاست؟")
    
    col_map1, col_map2 = st.columns(2)
    
    with col_map1:
        # Treemap of Risk
        fig_tree = px.treemap(
            df, path=['Department', 'Emp_ID'], values='Churn_Prob',
            color='Churn_Prob', color_continuous_scale='RdGy_r',
            title="نقشه ریسک سازمانی (کلیک کنید تا به سطح فرد برسید)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        
    with col_map2:
        st.markdown("#### 🚨 لیست تماشا (Watch List) - نخبگان در خطر")
        st.caption("۲۰ نفر برتر با بالاترین ریسک خروج و تخصص بالا (پتانسیل مهاجرت)")
        
        top_risk = df.sort_values(by='Churn_Prob', ascending=False).head(10)
        st.dataframe(
            top_risk[['Emp_ID', 'Department', 'Churn_Prob', 'Contract_Breach_Index', 'Migration_Risk']],
            column_config={
                "Churn_Prob": st.column_config.ProgressColumn("احتمال خروج", format="%.1f%%", min_value=0, max_value=100),
                "Migration_Risk": st.column_config.CheckboxColumn("ریسک مهاجرت"),
                "Contract_Breach_Index": st.column_config.NumberColumn("شاخص نارضایتی (0-10)")
            },
            hide_index=True
        )

# --- TAB 3: INTERVENTION (Strategy) ---
with tab3:
    st.subheader("سناریوهای مداخله: از داده تا درمان")
    st.markdown("بر اساس **فصل ششم گزارش**، کدام استراتژی برای سازمان شما به‌صرفه‌تر است؟")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.markdown("#### 🛠️ شبیه‌ساز بازآفرینی شغلی (Job Crafting)")
        st.info("اگر به جای افزایش حقوق (که بودجه نداریم)، 'استقلال کاری' و 'حمایت مدیر' را افزایش دهیم چه می‌شود؟")
        
        support_boost = st.slider("افزایش حمایت مدیران (آموزش منتورینگ)", 0, 50, 20, format="+%d%%")
        autonomy_boost = st.slider("افزایش استقلال و تفویض اختیار", 0, 50, 10, format="+%d%%")
        
        # Simulation Logic
        new_resources = df['Total_Resources'] * (1 + (support_boost + autonomy_boost)/100)
        new_risk_score = (df['Total_Demands'] * 0.3) - (new_resources * 0.3) + (df['Contract_Breach_Index'] * 0.4)
        new_churn_prob = ((new_risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())) * 100
        
        saved_employees = len(df[df['Churn_Prob'] > risk_threshold]) - len(df[new_churn_prob > risk_threshold])
        saved_cost = saved_employees * 0.3 # Billion Tomans
        
        st.success(f"""
        **نتیجه شبیه‌سازی:**
        با اجرای این طرح، ریسک خروج **{saved_employees} نفر** از وضعیت بحرانی خارج می‌شود.
        💰 **صرفه‌جویی مالی:** {saved_cost:.1f} میلیارد تومان (عدم نیاز به جذب نیروی جایگزین).
        """)
        
    with col_sim2:
        st.markdown("#### 🗣️ پروتکل مصاحبه ماندگاری (Stay Interview)")
        st.write("پیشنهاد سیستم برای نفرات لیست تماشا:")
        st.markdown("""
        1. **شفافیت مالی رادیکال:** توضیح صادقانه محدودیت‌های بودجه به تیم IT.
        2. **مداخله سطح ۲ (بازآفرینی):** برگزاری کارگاه برای تیم R&D جهت تطبیق علایق شخصی با پروژه.
        3. **جبران خدمات کل (Total Rewards):** ارائه وام یا پکیج‌های غیرنقدی برای کاهش اثر تورم بر تیم Operations.
        """)
