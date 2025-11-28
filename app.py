import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# --- 1. تنظیمات صفحه و استایل ---
st.set_page_config(
    page_title="سیستم هوشمند تحلیل پرسنل",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل CSS (فونت وزیر + راست‌چین + دیزاین مدرن)
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    
    * { font-family: 'Vazirmatn', sans-serif !important; }
    
    .stApp { background-color: #f4f6f9; }
    
    /* تنظیمات RTL */
    .main .block-container { direction: rtl; text-align: right; padding-top: 2rem; }
    .stSidebar { direction: rtl; text-align: right; }
    
    /* کارت‌های متریک */
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border-right: 5px solid #4c6ef5;
    }
    
    /* استایل جداول */
    .stDataFrame { direction: rtl; }
    
    /* باکس هوش مصنوعی */
    .ai-box {
        background-color: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-right: 5px solid #6366f1;
    }
    .ai-title { color: #4338ca; font-weight: bold; font-size: 1.2rem; display: flex; align-items: center; gap: 10px; }
    
</style>
""", unsafe_allow_html=True)

# --- 2. تولید داده‌های پیشرفته (شامل پرسش‌نامه و پیش‌بینی) ---
@st.cache_data
def generate_complex_data():
    np.random.seed(42)
    n = 300
    
    depts = ['فنی و مهندسی', 'فروش و بازاریابی', 'منابع انسانی', 'مالی', 'پشتیبانی']
    roles = ['کارشناس', 'مدیر میانی', 'مدیر ارشد', 'تکنسین']
    
    df = pd.DataFrame({
        'ID': range(1001, 1001 + n),
        'Name': [f"کارمند {i}" for i in range(1, n+1)],
        'Department': np.random.choice(depts, n),
        'Role': np.random.choice(roles, n),
        'Age': np.random.randint(22, 55, n),
        'Tenure': np.random.randint(1, 15, n), # سابقه کار
        
        # --- داده‌های پرسش‌نامه‌ای (Survey) ---
        'WorkLifeBalance': np.random.randint(1, 6, n), # 1 (بد) تا 5 (عالی)
        'ManagerSupport': np.random.randint(1, 6, n),
        'SalarySatisfaction': np.random.randint(1, 6, n),
        'CareerGrowth': np.random.randint(1, 6, n),
    })
    
    # --- محاسبات تحلیلی (Simulated AI Logic) ---
    
    # 1. محاسبه فرسودگی شغلی (Burnout): معکوس تعادل کار و زندگی + فشار مدیریت
    # فرمول: (6 - تعادل) * 0.5 + (6 - حمایت مدیر) * 0.5 (نتیجه بین 1 تا 5)
    df['BurnoutScore'] = ((6 - df['WorkLifeBalance']) * 0.6 + (6 - df['ManagerSupport']) * 0.4).round(1)
    
    # 2. احتمال مهاجرت (Migration Probability): سن پایین + رشد کم + تخصص بالا
    # عددی بین 0 تا 100
    df['MigrationProb'] = np.where(
        (df['Age'] < 35) & (df['CareerGrowth'] < 3), 
        np.random.randint(60, 95, n), # احتمال زیاد
        np.random.randint(10, 50, n)  # احتمال کم
    )
    
    # 3. قابلیت جایگزینی (Replaceability): نقش‌های پایین راحت‌تر جایگزین می‌شوند
    df['Replaceability'] = np.where(
        df['Role'].isin(['مدیر ارشد', 'مدیر میانی']), 
        'دشوار', 
        np.where(df['Role'] == 'کارشناس', 'متوسط', 'آسان')
    )
    
    return df

df = generate_complex_data()

# --- 3. سایدبار و فیلترها ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=70)
    st.header("پنل کنترل")
    st.markdown("---")
    
    selected_dept = st.multiselect(
        "فیلتر دپارتمان", options=df['Department'].unique(), default=df['Department'].unique()
    )
    
    min_burnout = st.slider(
        "حداقل نمره فرسودگی", 1.0, 5.0, 1.0, step=0.1,
        help="نمایش افرادی که نمره فرسودگی آن‌ها بالاتر از این مقدار است"
    )
    
    high_risk_only = st.checkbox("فقط نمایش ریسک مهاجرت بالا")

# اعمال فیلتر
df_filtered = df[df['Department'].isin(selected_dept)]
df_filtered = df_filtered[df_filtered['BurnoutScore'] >= min_burnout]

if high_risk_only:
    df_filtered = df_filtered[df_filtered['MigrationProb'] > 70]

# --- 4. بدنه اصلی ---
st.title("📊 داشبورد هوشمند سرمایه انسانی")
st.markdown(f"تعداد پرسنل انتخاب شده: **{len(df_filtered)} نفر**")
st.markdown("---")

# تعریف تب‌ها
tab1, tab2, tab3 = st.tabs(["📝 گزارش پرسش‌نامه‌ها", "⚠️ تحلیل ریسک و فرسودگی", "🤖 توصیه‌های هوشمند (AI)"])

# --- تب 1: داده‌ها و پرسش‌نامه‌ها ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("داده‌های خام و نتایج نظرسنجی")
        st.dataframe(
            df_filtered[['Name', 'Department', 'WorkLifeBalance', 'ManagerSupport', 'SalarySatisfaction']],
            use_container_width=True, height=400
        )
    with col2:
        st.subheader("توزیع رضایت شغلی")
        # میانگین امتیازات
        avg_scores = df_filtered[['WorkLifeBalance', 'ManagerSupport', 'SalarySatisfaction', 'CareerGrowth']].mean().reset_index()
        avg_scores.columns = ['شاخص', 'امتیاز']
        
        # دیکشنری ترجمه
        labels = {
            'WorkLifeBalance': 'تعادل کار/زندگی',
            'ManagerSupport': 'حمایت مدیر',
            'SalarySatisfaction': 'رضایت حقوق',
            'CareerGrowth': 'رشد شغلی'
        }
        avg_scores['شاخص'] = avg_scores['شاخص'].map(labels)
        
        fig_radar = px.line_polar(
            avg_scores, r='امتیاز', theta='شاخص', line_close=True,
            range_r=[0, 5], title="نمای کلی رضایت سازمانی"
        )
        fig_radar.update_layout(font_family="Vazirmatn")
        st.plotly_chart(fig_radar, use_container_width=True)

# --- تب 2: تحلیل فرسودگی و مهاجرت ---
with tab2:
    # KPI های این بخش
    k1, k2, k3 = st.columns(3)
    
    high_burnout_count = len(df_filtered[df_filtered['BurnoutScore'] > 4])
    potential_migrants = len(df_filtered[df_filtered['MigrationProb'] > 75])
    hard_to_replace = len(df_filtered[df_filtered['Replaceability'] == 'دشوار'])
    
    k1.metric("پرسنل دچار فرسودگی شدید", f"{high_burnout_count} نفر", delta_color="inverse")
    k2.metric("ریسک بالای مهاجرت", f"{potential_migrants} نفر", delta_color="inverse")
    k3.metric("پرسنل کلیدی (جایگزینی سخت)", f"{hard_to_replace} نفر", delta_color="normal")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("رابطه فرسودگی و احتمال مهاجرت")
        fig_scatter = px.scatter(
            df_filtered, x="BurnoutScore", y="MigrationProb",
            color="Department", size="SalarySatisfaction",
            hover_data=['Name', 'Role'],
            labels={'BurnoutScore': 'نمره فرسودگی (۱-۵)', 'MigrationProb': 'احتمال مهاجرت (%)'},
            title="آیا فرسودگی باعث مهاجرت می‌شود؟"
        )
        fig_scatter.update_layout(font_family="Vazirmatn", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c2:
        st.subheader("وضعیت جایگزینی پرسنل")
        fig_bar = px.histogram(
            df_filtered, x="Department", color="Replaceability",
            barmode="group",
            color_discrete_map={'دشوار': '#ff6b6b', 'متوسط': '#fcc419', 'آسان': '#51cf66'},
            title="سختی جایگزینی نیروها در هر دپارتمان"
        )
        fig_bar.update_layout(font_family="Vazirmatn", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- تب 3: توصیه‌های هوش مصنوعی (AI Recommendations) ---
with tab3:
    st.header("🤖 دستیار هوشمند تحلیل منابع انسانی")
    st.caption("این بخش با تحلیل داده‌های فیلتر شده، راهکارهای مدیریتی پیشنهاد می‌دهد.")
    
    # دکمه تولید تحلیل
    if st.button("تحلیل و تولید راهکار توسط AI"):
        with st.spinner("هوش مصنوعی در حال تحلیل داده‌ها..."):
            time.sleep(1.5) # شبیه‌سازی تاخیر پردازش
            
            # --- منطق تولید متن (Simulated AI) ---
            avg_burnout = df_filtered['BurnoutScore'].mean()
            avg_mig = df_filtered['MigrationProb'].mean()
            dominant_dept = df_filtered['Department'].mode()[0] if not df_filtered.empty else "کل سازمان"
            
            recommendations = []
            
            # تحلیل فرسودگی
            if avg_burnout > 3.5:
                recommendations.append(f"🔴 **هشدار فرسودگی:** میانگین نمره فرسودگی در {dominant_dept} بالاست ({avg_burnout:.1f}). پیشنهاد می‌شود طرح دورکاری یا کاهش ساعات کاری اجباری برای یک دوره کوتاه اجرا شود.")
            elif avg_burnout > 2.5:
                recommendations.append(f"🟡 **توجه:** سطح استرس در {dominant_dept} متوسط است. برگزاری کارگاه‌های مدیریت استرس توصیه می‌شود.")
            else:
                recommendations.append(f"🟢 **وضعیت خوب:** سطح انرژی و انگیزه در {dominant_dept} مطلوب است.")

            # تحلیل مهاجرت
            if avg_mig > 60:
                recommendations.append(f"✈️ **ریسک خروج:** احتمال مهاجرت یا ترک کار در این گروه بسیار بالاست. بررسی کنید آیا حقوق پرداختی با تورم و بازار کار همخوانی دارد؟ پیشنهاد می‌شود جلسات Stay Interview (مصاحبه ماندگاری) با افراد کلیدی برگزار شود.")
            
            # تحلیل جایگزینی
            if hard_to_replace > 5:
                recommendations.append(f"🔑 **مدیریت دانش:** شما {hard_to_replace} نیروی کلیدی دارید که جایگزینی آن‌ها دشوار است. آیا سیستم مستندسازی دانش (Knowledge Management) برای این افراد فعال است؟")

            # نمایش خروجی
            st.markdown(f"""
            <div class="ai-box">
                <div class="ai-title">💡 گزارش تحلیلی هوشمند</div>
                <br>
                <ul>
                    {''.join([f'<li style="margin-bottom:10px;">{rec}</li>' for rec in recommendations])}
                </ul>
                <hr>
                <div style="font-size:0.9rem; color:#666;">
                    <b>پیشنهاد اقدام فوری:</b><br>
                    با توجه به داده‌ها، اولویت اصلی شما باید <u>{'کاهش فرسودگی' if avg_burnout > 3 else 'حفظ نیروهای کلیدی'}</u> باشد.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("برای دریافت تحلیل هوشمند روی دکمه بالا کلیک کنید.")
        
    # بخش چت بات (ظاهری)
    st.markdown("### 💬 سوالات متداول از هوش مصنوعی")
    with st.expander("چگونه نرخ فرسودگی را کاهش دهم؟"):
        st.write("بر اساس داده‌های فعلی، مهمترین عامل فرسودگی 'عدم تعادل کار و زندگی' است. اصلاح ساعت‌های جلسات و جلوگیری از تماس‌های کاری در روزهای تعطیل می‌تواند موثر باشد.")
