import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. تنظیمات صفحه ---
st.set_page_config(
    page_title="داشبورد تحلیل منابع انسانی",
    page_icon="🏢",
    layout="wide"
)

# استایل برای راست‌چین کردن متون
st.markdown("""
<style>
    .main, .stSidebar { direction: rtl; text-align: right; }
    h1, h2, h3, h4, p, div, span { font-family: 'Tahoma', sans-serif; }
    .stMetric { text-align: right; }
</style>
""", unsafe_allow_html=True)

# --- 2. تولید داده‌های ساختگی (Mock Data) ---
@st.cache_data
def load_data():
    # شبیه‌سازی دیتاست IBM HR Analytics
    np.random.seed(42)
    n_employees = 500
    
    departments = ['فروش', 'تحقیق و توسعه', 'منابع انسانی']
    education_fields = ['پزشکی', 'علوم انسانی', 'فنی مهندسی', 'بازاریابی', 'سایر']
    job_roles = ['مدیر فروش', 'محقق', 'تکنسین آزمایشگاه', 'مدیر تولید', 'نماینده فروش', 'مدیر منابع انسانی']
    
    data = {
        'EmployeeID': range(1001, 1001 + n_employees),
        'Age': np.random.randint(22, 60, n_employees),
        'Gender': np.random.choice(['مرد', 'زن'], n_employees),
        'Department': np.random.choice(departments, n_employees),
        'EducationField': np.random.choice(education_fields, n_employees),
        'JobRole': np.random.choice(job_roles, n_employees),
        'MaritalStatus': np.random.choice(['مجرد', 'متحل', 'مطلقه'], n_employees),
        'YearsAtCompany': np.random.randint(1, 40, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 15, n_employees),
        'PerformanceRating': np.random.randint(1, 5, n_employees), # 1 (کم) تا 4 (عالی)
        'YearsInCurrentRole': np.random.randint(1, 15, n_employees),
        'MonthlyIncome': np.random.randint(3000, 20000, n_employees), # دلار
        'Attrition': np.random.choice(['Yes', 'No'], n_employees, p=[0.16, 0.84]) # 16% نرخ ریزش
    }
    
    df = pd.DataFrame(data)
    
    # محاسبه ستون‌های محاسباتی طبق قوانین مخزن گیت‌هاب
    # قانون ارتقا: اگر سال‌های پس از آخرین ارتقا >= 5 و عملکرد > 3 باشد (مثال)
    df['DueForPromotion'] = np.where(
        (df['YearsSinceLastPromotion'] >= 5) & (df['PerformanceRating'] >= 3), 
        'Yes', 'No'
    )
    
    # قانون تعدیل نیرو (Retrenchment) فرضی
    df['OnRetrenchmentList'] = np.where(
        (df['PerformanceRating'] <= 1) & (df['YearsAtCompany'] < 2),
        'Yes', 'No'
    )
    
    return df

df = load_data()

# --- 3. سایدبار و فیلترها ---
st.sidebar.header("🎛 فیلترهای سراسری")

# فیلتر دپارتمان
dept_filter = st.sidebar.multiselect(
    "انتخاب دپارتمان:",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

# فیلتر جنسیت
gender_filter = st.sidebar.multiselect(
    "انتخاب جنسیت:",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

# اعمال فیلتر
df_selection = df.query("Department == @dept_filter & Gender == @gender_filter")

if df_selection.empty:
    st.warning("داده‌ای با این فیلترها موجود نیست!")
    st.stop()

# --- 4. بدنه اصلی ---
st.title("🏢 داشبورد تحلیلی منابع انسانی (HR)")
st.markdown("تحلیل نیروی کار، نرخ ارتقا و ریزش نیرو بر اساس داده‌های سازمانی.")

# تب‌بندی مشابه پروژه اصلی
tab1, tab2, tab3 = st.tabs(["📊 خلاصه مدیریتی", "🚀 ظرفیت و ارتقا", "⚠️ تحلیل ریزش (Attrition)"])

# --- تب 1: خلاصه مدیریتی ---
with tab1:
    st.header("نمای کلی سازمان")
    
    # KPI ها
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("تعداد کل کارکنان", df_selection.shape[0])
    col2.metric("میانگین سنی", f"{int(df_selection['Age'].mean())} سال")
    col3.metric("میانگین حقوق", f"${int(df_selection['MonthlyIncome'].mean()):,}")
    col4.metric("نرخ ریزش کل", f"{round((df_selection[df_selection['Attrition']=='Yes'].shape[0] / df_selection.shape[0])*100, 1)}%")
    
    st.markdown("---")
    
    # نمودارهای سطر اول
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("توزیع جنسیتی در دپارتمان‌ها")
        fig_gender = px.histogram(
            df_selection, x="Department", color="Gender", 
            barmode="group", text_auto=True,
            color_discrete_map={'مرد': '#636EFA', 'زن': '#EF553B'},
            title="تعداد کارکنان به تفکیک دپارتمان و جنسیت"
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        
    with c2:
        st.subheader("توزیع سنی و تاهل")
        fig_age = px.box(
            df_selection, x="MaritalStatus", y="Age", color="MaritalStatus",
            title="پراکندگی سنی بر اساس وضعیت تاهل"
        )
        st.plotly_chart(fig_age, use_container_width=True)

# --- تب 2: ظرفیت و ارتقا ---
with tab2:
    st.header("تحلیل ارتقا شغلی و تعدیل")
    
    # محاسبه متریک‌های این بخش
    promo_count = df_selection[df_selection['DueForPromotion'] == 'Yes'].shape[0]
    retrench_count = df_selection[df_selection['OnRetrenchmentList'] == 'Yes'].shape[0]
    
    kpi1, kpi2 = st.columns(2)
    kpi1.metric("کاندیدای ارتقا شغلی (واجد شرایط)", promo_count, delta="نیاز به اقدام", delta_color="normal")
    kpi2.metric("لیست بررسی تعدیل (عملکرد پایین)", retrench_count, delta="خطر", delta_color="inverse")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # نمودار دایره‌ای کاندیدای ارتقا
        df_promo = df_selection.groupby('DueForPromotion').size().reset_index(name='Count')
        fig_promo = px.pie(
            df_promo, values='Count', names='DueForPromotion', 
            title="درصد کارکنان واجد شرایط ارتقا",
            color='DueForPromotion',
            color_discrete_map={'Yes': '#00CC96', 'No': '#EF553B'}
        )
        st.plotly_chart(fig_promo, use_container_width=True)
        
    with c2:
        # نمودار میله‌ای عملکرد بر اساس سال‌های حضور
        fig_perf = px.scatter(
            df_selection, x="YearsSinceLastPromotion", y="PerformanceRating",
            color="Department", size="MonthlyIncome",
            title="رابطه آخرین ارتقا و عملکرد (حباب = درآمد)"
        )
        st.plotly_chart(fig_perf, use_container_width=True)

# --- تب 3: تحلیل ریزش نیرو ---
with tab3:
    st.header("عوامل ترک سازمان")
    
    # فیلتر کردن فقط کسانی که رفته‌اند
    attrition_df = df_selection[df_selection['Attrition'] == 'Yes']
    
    if attrition_df.empty:
        st.success("هیچ ریزش نیرویی با فیلترهای فعلی یافت نشد!")
    else:
        st.markdown("تحلیل ویژگی‌های کارکنانی که سازمان را ترک کرده‌اند.")
        
        row1_1, row1_2 = st.columns(2)
        
        with row1_1:
            fig_att_dept = px.histogram(
                attrition_df, y="Department", x="Age", color="Gender",
                title="ریزش بر اساس دپارتمان و سن"
            )
            st.plotly_chart(fig_att_dept, use_container_width=True)
            
        with row1_2:
            fig_att_role = px.bar(
                attrition_df.groupby('JobRole').size().reset_index(name='Count'),
                x='Count', y='JobRole', orientation='h',
                title="کدام نقش‌های شغلی بیشترین ریزش را دارند؟"
            )
            st.plotly_chart(fig_att_role, use_container_width=True)
            
        # هیت‌مپ همبستگی (ساده شده)
        st.subheader("توزیع درآمد و سابقه کار در افراد جدا شده")
        fig_scatter_att = px.scatter(
            attrition_df, x="YearsAtCompany", y="MonthlyIncome",
            color="EducationField",
            title="درآمد در مقابل سابقه کار (افراد جدا شده)"
        )
        st.plotly_chart(fig_scatter_att, use_container_width=True)
