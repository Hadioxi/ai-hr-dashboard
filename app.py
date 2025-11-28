import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# --- 1. تنظیمات صفحه ---
st.set_page_config(
    page_title="داشبورد جامع سرمایه انسانی",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. استایل‌دهی حرفه‌ای (CSS) ---
# حل مشکل فونت سفید + طراحی کارت‌های مدیریتی
st.markdown("""
<style>
    /* ایمپورت فونت وزیر */
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    
    * {
        font-family: 'Vazirmatn', sans-serif !important;
        color: #1f2937; /* رنگ متن پیش‌فرض: خاکستری تیره */
    }
    
    /* اجبار پس‌زمینه روشن برای کل اپ */
    .stApp {
        background-color: #f3f4f6;
    }
    
    /* تنظیمات RTL */
    .main .block-container {
        direction: rtl;
        text-align: right;
        padding-top: 1rem;
    }
    
    /* استایل سایدبار */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-left: 1px solid #e5e7eb;
    }
    
    section[data-testid="stSidebar"] * {
        color: #1f2937 !important; /* متن سایدبار همیشه مشکی */
    }

    /* کارت‌های KPI (شاخص‌های کلیدی) */
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-right: 5px solid #3b82f6; /* نوار آبی سمت راست */
        text-align: right;
        margin-bottom: 10px;
    }
    .kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 5px; }
    .kpi-value { font-size: 1.8rem; font-weight: bold; color: #111827; }
    .kpi-delta { font-size: 0.8rem; color: #10b981; } /* رنگ سبز */
    .kpi-delta.neg { color: #ef4444; } /* رنگ قرمز */

    /* استایل تب‌ها */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: flex-end;
        border-bottom: 2px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* باکس هوش مصنوعی */
    .ai-insight-box {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
</style>
""", unsafe_allow_html=True)

# --- 3. تولید داده‌های پیشرفته ---
@st.cache_data
def generate_executive_data():
    np.random.seed(42)
    n = 500 # تعداد پرسنل
    
    depts = ['فنی و مهندسی', 'فروش', 'مالی', 'پشتیبانی', 'R&D']
    
    df = pd.DataFrame({
        'ID': range(1001, 1001 + n),
        'Department': np.random.choice(depts, n),
        
        # فیلترهای مورد نظر شما
        'Salary': np.random.randint(13, 85, n) * 1000000, # حقوق (۱۳ تا ۸۵ میلیون)
        'MonthlyHours': np.random.normal(176, 20, n).astype(int), # میانگین ۱۷۶ ساعت (استاندارد)
        'Tenure': np.random.randint(1, 20, n), # سابقه کار
        
        # شاخص‌های کیفی
        'Satisfaction': np.random.randint(1, 10, n),
        'ManagerRating': np.random.randint(1, 6, n),
        'Age': np.random.randint(23, 60, n)
    })
    
    # اطمینان از اینکه ساعات کاری زیر ۱۰۰ یا بالای ۲۵۰ نیست (داده پرت)
    df['MonthlyHours'] = df['MonthlyHours'].clip(120, 260)
    
    # --- منطق تحلیلی (Calculated Fields) ---
    
    # 1. محاسبه ریسک فرسودگی (Burnout Risk)
    # اگر ساعات کاری بالا باشد (>200) و رضایت پایین باشد
    conditions = [
        (df['MonthlyHours'] > 200) & (df['Satisfaction'] < 5),
        (df['MonthlyHours'] > 180) | (df['Satisfaction'] < 7),
    ]
    choices = ['خطرناک', 'هشدار']
    df['BurnoutStatus'] = np.select(conditions, choices, default='نرمال')
    
    # 2. احتمال مهاجرت (Migration Probability)
    # متخصصین جوان (سن < 35) با حقوق نسبتاً پایین نسبت به بازار (مثلا < 30 میلیون)
    df['MigrationProb'] = np.where(
        (df['Age'] < 35) & (df['Salary'] < 30000000) & (df['Department'].isin(['فنی و مهندسی', 'R&D'])),
        np.random.randint(70, 99, n), # درصد بالا
        np.random.randint(10, 50, n)
    )
    
    # 3. قابلیت جایگزینی (Replaceability)
    # سابقه بالا = سخت
    df['Replaceability'] = np.where(df['Tenure'] > 7, 'سخت', np.where(df['Tenure'] > 3, 'متوسط', 'آسان'))
    
    return df

df_full = generate_executive_data()

# --- 4. سایدبار (پنل کنترل مدیر) ---
with st.sidebar:
    st.markdown("### ⚙️ فیلترهای پیشرفته")
    st.markdown("---")
    
    # فیلتر دپارتمان
    sel_dept = st.multiselect(
        "واحد سازمانی",
        options=df_full['Department'].unique(),
        default=df_full['Department'].unique()
    )
    
    # فیلتر 1: حداقل حقوق (درخواست کاربر)
    st.markdown("**💰 محدوده حقوق (تومان)**")
    min_sal, max_sal = st.slider(
        "بازه حقوقی",
        min_value=13000000, 
        max_value=100000000, 
        value=(13000000, 85000000),
        step=1000000,
        format="%d"
    )
    
    # فیلتر 2: ساعات کاری (درخواست کاربر)
    st.markdown("**⏰ ساعات کاری ماهانه**")
    hours_range = st.slider(
        "فیلتر ساعت کاری",
        min_value=120,
        max_value=260,
        value=(140, 220),
        help="استاندارد: ۱۷۶ ساعت"
    )
    
    # فیلتر 3: فاکتورهای دیگر (سابقه کار)
    st.markdown("**📅 سابقه کار (سال)**")
    tenure_min = st.slider("حداقل سابقه", 0, 20, 0)
    
    st.markdown("---")
    st.caption("v2.1.0 | داشبورد مدیریتی")

# --- اعمال فیلترها ---
df = df_full[
    (df_full['Department'].isin(sel_dept)) &
    (df_full['Salary'] >= min_sal) & (df_full['Salary'] <= max_sal) &
    (df_full['MonthlyHours'] >= hours_range[0]) & (df_full['MonthlyHours'] <= hours_range[1]) &
    (df_full['Tenure'] >= tenure_min)
]

# --- 5. بدنه اصلی داشبورد ---

# هدر
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.title("سامانه تحلیل استراتژیک سرمایه انسانی")
    st.markdown(f"**فیلتر فعال:** پرسنل با حقوق بالای {min_sal:,.0f} تومان و ساعات کاری بین {hours_range[0]} تا {hours_range[1]}")

with c_head2:
    # دکمه دانلود (نمادین)
    st.button("📥 دانلود گزارش اکسل", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ردیف KPI سفارشی (با HTML برای زیبایی)
col1, col2, col3, col4 = st.columns(4)

# محاسبات KPI
avg_sal = df['Salary'].mean() / 1000000 if not df.empty else 0
avg_hours = df['MonthlyHours'].mean() if not df.empty else 0
high_risk_burnout = len(df[df['BurnoutStatus'] == 'خطرناک'])
mig_risk_count = len(df[df['MigrationProb'] > 80])

def kpi_card(title, value, delta, color="green"):
    delta_cls = "neg" if color == "red" else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_cls}">{delta}</div>
    </div>
    """

with col1:
    st.markdown(kpi_card("میانگین حقوق پرداختی", f"{avg_sal:,.1f} M", "تومان"), unsafe_allow_html=True)
with col2:
    color = "red" if avg_hours > 190 else "green"
    st.markdown(kpi_card("میانگین ساعات کاری", f"{int(avg_hours)}", "ساعت/ماه", color), unsafe_allow_html=True)
with col3:
    st.markdown(kpi_card("پرسنل در خطر فرسودگی", f"{high_risk_burnout}", "نفر (نیاز به توجه)", "red"), unsafe_allow_html=True)
with col4:
    st.markdown(kpi_card("ریسک مهاجرت قطعی", f"{mig_risk_count}", "نیروی کلیدی", "red"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# تب‌ها
tab_risk, tab_survey, tab_ai = st.tabs(["⚠️ تحلیل ریسک و فرسودگی", "📊 تحلیل جمعیت‌شناسی", "🤖 توصیه‌های هوشمند (AI)"])

# --- تب 1: تحلیل ریسک (تمرکز اصلی مدیران) ---
with tab_risk:
    r1, r2 = st.columns([2, 1])
    
    with r1:
        st.subheader("رابطه فشار کاری و فرسودگی")
        # نمودار Scatter
        fig_burn = px.scatter(
            df, x="MonthlyHours", y="Satisfaction",
            color="BurnoutStatus",
            size="Salary", # اندازه حباب = حقوق
            hover_data=['ID', 'Department'],
            color_discrete_map={'خطرناک': '#ef4444', 'هشدار': '#f59e0b', 'نرمال': '#10b981'},
            labels={'MonthlyHours': 'ساعات کاری ماهانه', 'Satisfaction': 'رضایت شغلی (۱-۱۰)'},
            title="تحلیل پراکندگی: ساعات کاری vs رضایت"
        )
        fig_burn.update_layout(font_family="Vazirmatn", plot_bgcolor="white", paper_bgcolor="white")
        fig_burn.update_xaxes(showgrid=True, gridcolor='#f3f4f6')
        fig_burn.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
        st.plotly_chart(fig_burn, use_container_width=True)
        
    with r2:
        st.subheader("قابلیت جایگزینی پرسنل")
        # نمودار دونات
        fig_rep = px.pie(
            df, names='Replaceability', 
            hole=0.6,
            color='Replaceability',
            color_discrete_map={'سخت': '#ef4444', 'متوسط': '#f59e0b', 'آسان': '#10b981'}
        )
        fig_rep.update_layout(font_family="Vazirmatn", showlegend=False, 
                              annotations=[dict(text=f'{len(df)}', x=0.5, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig_rep, use_container_width=True)
        st.caption("تعداد کل پرسنل در مرکز نمودار")

# --- تب 2: جمعیت شناسی ---
with tab_survey:
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("توزیع حقوق در دپارتمان‌ها")
        fig_box = px.box(
            df, x="Department", y="Salary", color="Department",
            title="پراکندگی حقوق (باکس‌پلات)"
        )
        fig_box.update_layout(font_family="Vazirmatn", showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with d2:
        st.subheader("هرم سابقه کار")
        fig_hist = px.histogram(
            df, x="Tenure", nbins=10, 
            title="توزیع فراوانی سابقه کار پرسنل",
            color_discrete_sequence=['#3b82f6']
        )
        fig_hist.update_layout(font_family="Vazirmatn", plot_bgcolor="white", bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

# --- تب 3: هوش مصنوعی (AI) ---
with tab_ai:
    # دکمه اجرای تحلیل
    if st.button("🧠 بازخوانی و تحلیل داده‌ها توسط هوش مصنوعی", type="primary"):
        with st.spinner("در حال پردازش الگوهای رفتاری و سازمانی..."):
            time.sleep(2)
            
            # منطق تولید متن هوشمند
            insights = []
            
            # تحلیل حقوق
            if min_sal > 20000000:
                insights.append(f"شما فیلتر حقوق را روی حداقل {min_sal:,.0f} تومان تنظیم کرده‌اید. در این سطح درآمدی، انتظار می‌رود 'رضایت شغلی' بالا باشد. اگر نمودارها رضایت پایین را نشان می‌دهند، مشکل **فرهنگ سازمانی** یا **مدیریت میکرومناجمنت** است، نه پول.")
            
            # تحلیل ساعات کاری
            if avg_hours > 185:
                insights.append(f"میانگین ساعات کاری ({int(avg_hours)} ساعت) بالاتر از استاندارد قانونی است. داده‌ها نشان می‌دهند {high_risk_burnout} نفر در وضعیت قرمز فرسودگی هستند. این زنگ خطری برای **افزایش خطاهای انسانی** و **سوانح کاری** است.")
            
            # تحلیل مهاجرت
            if mig_risk_count > (len(df) * 0.2):
                insights.append(f"بیش از ۲۰٪ نیروهای فیلتر شده ({mig_risk_count} نفر) پتانسیل بالای مهاجرت دارند. از آنجا که این افراد در بازه حقوقی انتخابی شما هستند، احتمالاً رقبای بین‌المللی با پیشنهاد **Work-Life Balance** بهتر آن‌ها را جذب می‌کنند.")

            # تحلیل جایگزینی
            hard_replace_perc = (len(df[df['Replaceability']=='سخت']) / len(df)) * 100
            if hard_replace_perc > 30:
                insights.append("سازمان وابستگی شدیدی به افراد قدیمی دارد. برنامه **جانشین‌پروری (Succession Planning)** باید فوراً برای پوزیشن‌های کلیدی اجرا شود.")

            # نمایش
            st.markdown(f"""
            <div class="ai-insight-box">
                <h3 style="color:#2563eb; display:flex; align-items:center;">
                    <span style="font-size:1.5rem; margin-left:10px;">🤖</span> گزارش تحلیلی مدیر عامل
                </h3>
                <p style="color:#4b5563; font-size:0.95rem; line-height:1.8;">
                    بر اساس فیلترهای اعمال شده (حقوق بالای {min_sal//1000000} میلیون و ساعات کاری {hours_range[0]}-{hours_range[1]}), هوش مصنوعی نکات زیر را استخراج کرد:
                </p>
                <ul style="color:#1f2937; font-weight:500; line-height:2;">
                    {''.join([f'<li>{item}</li>' for item in insights])}
                </ul>
                <div style="margin-top:20px; padding:10px; background:#dbeafe; border-radius:8px; color:#1e40af; font-size:0.9rem;">
                    <strong>💡 پیشنهاد استراتژیک:</strong> 
                    {'کاهش فشار کاری و اضافه کار اجباری' if avg_hours > 180 else 'بازنگری در پکیج‌های نگهداشت (Retention)'} را در اولویت قرار دهید.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("برای دریافت تحلیل جامع متنی، روی دکمه بالا کلیک کنید.")
