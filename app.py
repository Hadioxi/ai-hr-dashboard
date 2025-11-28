import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. تنظیمات صفحه و تم ---
st.set_page_config(
    page_title="داشبورد منابع انسانی",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. تزریق CSS حرفه‌ای (جادوی زیبایی) ---
st.markdown("""
<style>
    /* ایمپورت فونت وزیر */
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    
    * {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    
    /* تنظیمات اصلی بدنه و راست‌چین */
    .stApp {
        background-color: #f8f9fa;
    }
    
    .main .block-container {
        direction: rtl;
        padding-top: 2rem;
    }
    
    /* استایل سایدبار */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    
    /* کارت‌های متریک (KPI Cards) */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        text-align: right;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        border-color: #4c6ef5;
    }
    
    /* عناوین و متن‌ها */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
        text-align: right;
    }
    
    /* تب‌ها */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: flex-end;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 5px;
        color: #555;
        font-size: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4c6ef5 !important;
        color: white !important;
    }

    /* پنهان کردن منوی همبرگری پیش‌فرض برای ظاهر تمیزتر */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- 3. تابع تولید داده (همان منطق قبلی) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_employees = 500
    data = {
        'EmployeeID': range(1001, 1001 + n_employees),
        'Age': np.random.randint(22, 60, n_employees),
        'Gender': np.random.choice(['مرد', 'زن'], n_employees),
        'Department': np.random.choice(['فروش', 'تحقیق و توسعه', 'منابع انسانی', 'IT', 'مالی'], n_employees),
        'JobRole': np.random.choice(['مدیر', 'کارشناس ارشد', 'کارشناس', 'کارآموز'], n_employees),
        'MonthlyIncome': np.random.randint(15, 120, n_employees) * 1000000, # تومان
        'Attrition': np.random.choice(['بله', 'خیر'], n_employees, p=[0.16, 0.84]),
        'PerformanceRating': np.random.randint(1, 6, n_employees),
        'YearsAtCompany': np.random.randint(1, 20, n_employees)
    }
    df = pd.DataFrame(data)
    df['Status'] = np.where(df['Attrition'] == 'بله', 'ترک کار', 'فعال')
    return df

df = load_data()

# --- 4. سایدبار حرفه‌ای ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("پنل تنظیمات")
    st.markdown("---")
    
    # فیلترها با استایل بهتر
    st.subheader("📌 فیلترهای نمایش")
    
    selected_dept = st.multiselect(
        "دپارتمان‌ها",
        options=df['Department'].unique(),
        default=df['Department'].unique(),
        help="انتخاب یک یا چند دپارتمان"
    )
    
    selected_gender = st.multiselect(
        "جنسیت",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )
    
    st.markdown("---")
    st.info("💡 این داشبورد وضعیت منابع انسانی سازمان را بر اساس داده‌های سال ۱۴۰۳ نمایش می‌دهد.")

# اعمال فیلتر
df_filtered = df.query("Department == @selected_dept & Gender == @selected_gender")

# --- 5. بدنه اصلی ---

# هدر اصلی با طراحی متفاوت
c1, c2 = st.columns([1, 4])
with c2:
    st.title("داشبورد جامع منابع انسانی")
    st.markdown(f"🗓 **آخرین بروزرسانی:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
with c1:
    # نمایش تعداد کل به صورت خیلی بزرگ
    st.markdown(
        f"""
        <div style="background-color:#4c6ef5; color:white; padding:10px; border-radius:10px; text-align:center;">
            <div style="font-size:14px;">تعداد پرسنل</div>
            <div style="font-size:32px; font-weight:bold;">{len(df_filtered)}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# متریک‌های کلیدی (KPIs)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

avg_age = int(df_filtered['Age'].mean())
avg_income = int(df_filtered['MonthlyIncome'].mean() / 1000000)
attrition_rate = round((len(df_filtered[df_filtered['Attrition']=='بله']) / len(df_filtered)) * 100, 1)
avg_perf = round(df_filtered['PerformanceRating'].mean(), 1)

with kpi1:
    st.metric("میانگین سنی", f"{avg_age} سال", delta_color="off")
with kpi2:
    st.metric("میانگین حقوق", f"{avg_income} م.تومان", delta_color="off")
with kpi3:
    st.metric("نرخ ریزش نیرو", f"{attrition_rate}%", "-2%" if attrition_rate < 15 else "+1%")
with kpi4:
    st.metric("میانگین عملکرد (۱-۵)", f"{avg_perf}", "خوب" if avg_perf > 3 else "نیاز به بهبود")

st.markdown("---")

# --- تب‌ها با محتوای بصری ---
tab1, tab2 = st.tabs(["📊 تحلیل جمعیت‌شناسی", "⚠️ تحلیل ترک کار (Attrition)"])

# تابع کمکی برای استایل دادن به نمودارها
def beautify_plotly(fig):
    fig.update_layout(
        font_family="Vazirmatn",
        title_font_family="Vazirmatn",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#2c3e50"),
        margin=dict(t=50, l=10, r=10, b=10)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    return fig

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("توزیع پرسنل در دپارتمان‌ها")
        fig_dept = px.bar(
            df_filtered['Department'].value_counts().reset_index(),
            x='Department', y='count',
            color='count',
            text_auto=True,
            labels={'Department': 'دپارتمان', 'count': 'تعداد'},
            color_continuous_scale="Blues"
        )
        st.plotly_chart(beautify_plotly(fig_dept), use_container_width=True)
        
    with col_b:
        st.subheader("ترکیب جنسیتی و نقش‌ها")
        fig_sun = px.sunburst(
            df_filtered, path=['Gender', 'JobRole'],
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(beautify_plotly(fig_sun), use_container_width=True)

    st.subheader("روند حقوق و سابقه کار")
    fig_scatter = px.scatter(
        df_filtered, x="YearsAtCompany", y="MonthlyIncome",
        size="PerformanceRating", color="Department",
        hover_data=['JobRole'],
        labels={'YearsAtCompany': 'سابقه کار (سال)', 'MonthlyIncome': 'حقوق ماهیانه'},
        color_discrete_sequence=px.colors.qualitative.G10
    )
    st.plotly_chart(beautify_plotly(fig_scatter), use_container_width=True)

with tab2:
    col_c, col_d = st.columns([2, 1])
    
    with col_c:
        st.subheader("چه کسانی سازمان را ترک می‌کنند؟")
        attrition_data = df_filtered[df_filtered['Attrition'] == 'بله']
        
        if attrition_data.empty:
            st.success("هیچ داده‌ای برای نمایش وجود ندارد.")
        else:
            fig_att = px.histogram(
                attrition_data, x="Department", color="JobRole",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'Department': 'دپارتمان', 'count': 'تعداد خروج'}
            )
            st.plotly_chart(beautify_plotly(fig_att), use_container_width=True)
    
    with col_d:
        st.subheader("وضعیت کلی ریزش")
        fig_donut = px.pie(
            df_filtered, names='Attrition',
            hole=0.6,
            color_discrete_map={'بله': '#ff6b6b', 'خیر': '#51cf66'}
        )
        st.plotly_chart(beautify_plotly(fig_donut), use_container_width=True)
