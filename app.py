import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# 1. تنظیمات و ظاهر سیستم
# ---------------------------------------------------------
st.set_page_config(
    page_title="سیستم نبض‌سنج سازمانی | نسخه هوشمند",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل‌دهی: تمیز، مینیمال و متمرکز بر نواحی رنگی (قرمز، زرد، سبز)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Tahoma', sans-serif; color: #ffffff; }
    
    /* کارت‌های وضعیت */
    .zone-card { padding: 15px; border-radius: 8px; margin-bottom: 10px; color: white; text-align: center; }
    .zone-red { background-color: #7f1d1d; border: 2px solid #ef4444; }
    .zone-yellow { background-color: #78350f; border: 2px solid #f59e0b; }
    .zone-green { background-color: #064e3b; border: 2px solid #10b981; }
    
    .big-num { font-size: 2rem; font-weight: bold; }
    .desc { font-size: 0.9rem; opacity: 0.8; }
    
    /* جدول اقدامات */
    div[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. شبیه‌سازی داده‌ها (ورودی‌های میکرو-نظرسنجی + تردد)
# ---------------------------------------------------------
@st.cache_data
def load_pulse_data():
    np.random.seed(1403)
    n = 150 # تعداد پرسنل
    
    # داده‌های پایه
    ids = [f"P-{i+100}" for i in range(n)]
    names = [f"کارمند {i+1}" for i in range(n)]
    depts = np.random.choice(['فروش', 'فنی', 'منابع انسانی', 'عملیات'], n)
    is_elite = np.random.choice([True, False], n, p=[0.2, 0.8]) # ۲۰ درصد نخبه
    
    # 1. سوال اول: سنجش فشار (JD-R) - (1 کم، 10 زیاد)
    pressure_score = np.random.normal(6, 2, n).clip(1, 10)
    
    # 2. سوال دوم: سنجش قرارداد روانی (تعهد سازمان) - (1 کم، 10 زیاد)
    contract_score = np.random.normal(5, 2.5, n).clip(1, 10)
    
    # 3. داده‌های تردد (از سیستم حضور و غیاب)
    # تاخیر زیاد با رضایت کم همبستگی دارد
    lateness_avg = (10 - contract_score) * 5 + np.random.normal(0, 10, n)
    lateness_avg = lateness_avg.clip(0, 120) # دقیقه در ماه
    
    df = pd.DataFrame({
        'ID': ids,
        'Name': names,
        'Department': depts,
        'Is_Elite': is_elite,
        'Pressure_Score': pressure_score,   # فشار کار
        'Contract_Score': contract_score,   # احساس عدالت/وفای به عهد
        'Lateness_Minutes': lateness_avg    # رفتار (آژیر)
    })
    
    # --- موتور تصمیم‌ساز (منطبق بر لاجیک شما) ---
    def categorize(row):
        # ناحیه قرمز: فشار بالا + بی‌عدالتی + نخبه بودن (یا تاخیر زیاد که نشانه خطر است)
        if (row['Pressure_Score'] > 7 or row['Contract_Score'] < 4) and row['Is_Elite']:
            return "قرمز (بحرانی)"
        elif (row['Contract_Score'] < 4) and (row['Lateness_Minutes'] > 60):
             return "قرمز (بحرانی)"
             
        # ناحیه زرد: احساس نقض قرارداد (بی‌عدالتی) اما فشار متعادل
        elif row['Contract_Score'] < 6:
            return "زرد (استعفای خاموش)"
            
        # ناحیه سبز: همه چیز نرمال
        else:
            return "سبز (ایمن)"

    df['Zone'] = df.apply(categorize, axis=1)
    
    # تعیین اقدام (تجویز)
    def prescribe(row):
        if row['Zone'] == "قرمز (بحرانی)":
            return "مصاحبه ماندگاری (فوری)"
        elif row['Zone'] == "زرد (استعفای خاموش)":
            return "بازآفرینی شغلی + شفافیت"
        else:
            return "تشویق و حفظ وضعیت"
            
    df['Action'] = df.apply(prescribe, axis=1)
    
    return df

df = load_pulse_data()

# ---------------------------------------------------------
# 3. سایدبار (کنترل پنل)
# ---------------------------------------------------------
with st.sidebar:
    st.title("💓 نبض‌سنج سازمانی")
    st.write("رصد لحظه‌ای وضعیت روانی پرسنل")
    st.markdown("---")
    
    filter_dept = st.multiselect("فیلتر دپارتمان:", df['Department'].unique(), default=df['Department'].unique())
    filter_zone = st.multiselect("فیلتر وضعیت:", df['Zone'].unique(), default=["قرمز (بحرانی)", "زرد (استعفای خاموش)"])
    
    st.info("""
    **منطق سیستم:**
    🟢 **سبز:** تعادل برقرار است.
    🟡 **زرد:** استعفای خاموش (بی‌انگیزه).
    🔴 **قرمز:** خطر خروج قطعی (نیاز به اقدام فوری).
    """)

# اعمال فیلتر
df_filtered = df[df['Department'].isin(filter_dept) & df['Zone'].isin(filter_zone)]

# ---------------------------------------------------------
# 4. داشبورد اصلی
# ---------------------------------------------------------

st.title("داشبورد تحلیل و اقدام پیش‌دستانه")
st.markdown("این سیستم بر اساس داده‌های **میکرو-نظرسنجی ماهانه** و **رفتار تردد**، صدای شکستن تعهد کارکنان را می‌شنود.")

# --- بخش ۱: نمای کلی (کارت‌های رنگی) ---
col1, col2, col3 = st.columns(3)
red_count = len(df[df['Zone'] == "قرمز (بحرانی)"])
yellow_count = len(df[df['Zone'] == "زرد (استعفای خاموش)"])
green_count = len(df[df['Zone'] == "سبز (ایمن)"])

with col1:
    st.markdown(f"""
    <div class="zone-card zone-red">
        <div class="big-num">{red_count} نفر</div>
        <div class="desc">ناحیه قرمز (خطر مهاجرت/خروج)</div>
        <div class="desc">نخبگانی که فشار بالا و حس بی‌عدالتی دارند</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="zone-card zone-yellow">
        <div class="big-num">{yellow_count} نفر</div>
        <div class="desc">ناحیه زرد (استعفای خاموش)</div>
        <div class="desc">حضور فیزیکی دارند اما دلشان رفته است</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="zone-card zone-green">
        <div class="big-num">{green_count} نفر</div>
        <div class="desc">ناحیه سبز (پایدار)</div>
        <div class="desc">وضعیت مطلوب و متعادل</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- بخش ۲: تحلیل و تجویز (Actionable Insights) ---
tab_action, tab_analysis = st.tabs(["💊 اتاق درمان (اقدامات عملی)", "📊 نمودار تحلیل (ماتریس فشار-عدالت)"])

with tab_action:
    st.subheader("لیست اقدامات پیشنهادی (بدون بودجه کلان)")
    st.markdown("بر اساس وضعیت هر فرد، سیستم یکی از راهکارهای **مصاحبه ماندگاری**، **بازآفرینی شغلی** یا **شفافیت** را پیشنهاد می‌دهد.")
    
    # نمایش جدول رنگی
    def highlight_row(row):
        color = ''
        if 'قرمز' in row.Zone: color = 'background-color: #450a0a; color: #fecaca'
        elif 'زرد' in row.Zone: color = 'background-color: #422006; color: #fde68a'
        return color

    st.dataframe(
        df_filtered[['Name', 'Department', 'Zone', 'Lateness_Minutes', 'Action']].sort_values('Zone'),
        column_config={
            "Name": "نام پرسنل",
            "Department": "واحد",
            "Zone": "وضعیت (تشخیص)",
            "Lateness_Minutes": st.column_config.NumberColumn("دقایق تاخیر (رفتار)", format="%d min"),
            "Action": "نسخه تجویزی (اقدام مدیر)"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # راهنمای اقدام (توضیحات متنی مدل شما)
    with st.expander("راهنمای اجرای اقدامات (کلیک کنید)"):
        c1, c2 = st.columns(2)
        with c1:
            st.warning("### 🔴 برای ناحیه قرمز: مصاحبه ماندگاری")
            st.write("""
            **هدف:** شناسایی تنها مانعی که فرد را فراری می‌دهد.
            **سوال کلیدی:** «دقیقاً چه چیزی تو را اینجا نگه می‌دارد و چه چیزی تو را فراری می‌دهد؟»
            **اقدام:** رفع همان یک مانع (حتی اگر کوچک باشد).
            """)
        with c2:
            st.info("### 🟡 برای ناحیه زرد: بازآفرینی شغلی")
            st.write("""
            **هدف:** معنا بخشیدن به کار وقتی پول نیست.
            **دیالوگ:** «ما نمی‌توانیم حقوق را دو برابر کنیم، اما می‌توانیم شغل را آنطور که دوست داری تغییر دهیم.»
            **اقدام:** اجازه دهید بخشی از وظایف یا هم‌تیمی‌هایش را خودش انتخاب کند.
            """)

with tab_analysis:
    st.subheader("ماتریس تشخیص وضعیت")
    st.markdown("توزیع کارکنان بر اساس **فشار وارده (JD-R)** و **احساس عدالت (قرارداد روانی)**.")
    
    # Scatter Plot
    fig = px.scatter(
        df, x="Pressure_Score", y="Contract_Score", color="Zone",
        size="Lateness_Minutes", hover_data=['Name', 'Is_Elite'],
        color_discrete_map={
            "قرمز (بحرانی)": "#ef4444",
            "زرد (استعفای خاموش)": "#f59e0b",
            "سبز (ایمن)": "#10b981"
        },
        labels={"Pressure_Score": "فشار کار (JD-R)", "Contract_Score": "احساس عدالت (قرارداد روانی)"},
        template="plotly_dark", height=500
    )
    # خطوط راهنما
    fig.add_hline(y=4, line_dash="dot", line_color="white", annotation_text="مرز احساس بی‌عدالتی")
    fig.add_vline(x=7, line_dash="dot", line_color="white", annotation_text="مرز فرسودگی")
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("نکته: دایره‌های بزرگتر نشان‌دهنده تاخیر بیشتر (نشانه رفتاری نارضایتی) هستند.")
