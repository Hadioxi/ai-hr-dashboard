import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- تنظیمات صفحه ---
st.set_page_config(
    page_title="داشبورد پیش‌بینی خروج کارکنان",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- استایل‌دهی راست‌چین برای فارسی ---
st.markdown("""
<style>
    .main {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', sans-serif;
    }
    h1, h2, h3, h4 {
        text-align: right;
        font-family: 'Tahoma', sans-serif;
    }
    .stMetric {
        direction: rtl; 
        text-align: right;
    }
    /* تنظیم فونت نمودارها */
    .js-plotly-plot .plotly .g-title {
        font-family: 'Tahoma', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. تولید داده‌های ساختگی (Mock Data) ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_employees = 300
    
    # داده‌های دموگرافیک
    ids = [f"EMP-{i:03d}" for i in range(1, n_employees + 1)]
    departments = np.random.choice(['فنی و مهندسی', 'فروش و مارکتینگ', 'منابع انسانی', 'مالی', 'پشتیبانی'], n_employees)
    tenure = np.random.randint(1, 15, n_employees) # سابقه کار
    
    # شاخص‌های روانشناسی صنعتی سازمانی (نمره 1 تا 10)
    # هرچه نمره بالاتر، وضعیت بهتر (بجز فرسودگی)
    
    # تعهد عاطفی (علاقه به سازمان)
    affective_commitment = np.random.normal(6, 2, n_employees).clip(1, 10)
    
    # عدالت سازمانی (احساس انصاف)
    organizational_justice = np.random.normal(5.5, 2.5, n_employees).clip(1, 10)
    
    # کیفیت رابطه با مدیر (LMX)
    lmx = np.random.normal(6, 2, n_employees).clip(1, 10)
    
    # فرسودگی شغلی (Burnout) - نمره بالا یعنی فرسودگی بیشتر (بد)
    burnout = np.random.normal(4, 2.5, n_employees).clip(1, 10)
    
    # تناسب شغل و فرد (P-J Fit)
    job_fit = np.random.normal(7, 1.5, n_employees).clip(1, 10)

    df = pd.DataFrame({
        'ID': ids,
        'Department': departments,
        'Tenure_Years': tenure,
        'Commitment': affective_commitment,
        'Justice': organizational_justice,
        'LMX_Manager_Rel': lmx,
        'Burnout': burnout,
        'Job_Fit': job_fit
    })

    # ایجاد ستون هدف (احتمال خروج) بر اساس فرمول منطقی برای شبیه‌سازی واقعیت
    # فرمول: خروج بالا = فرسودگی بالا + عدالت پایین + تعهد پایین
    risk_score = (
        (df['Burnout'] * 1.5) + 
        ((11 - df['Justice']) * 1.2) + 
        ((11 - df['Commitment']) * 1.0) +
        ((11 - df['LMX_Manager_Rel']) * 0.8)
    )
    
    # نرمال‌سازی ریسک بین 0 تا 100
    df['Risk_Score'] = ((risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())) * 100
    
    # برچسب‌گذاری (اگر ریسک بالای 60 باشد، احتمال خروج بالاست)
    df['Will_Leave'] = (df['Risk_Score'] > 60).astype(int)
    
    return df

df = generate_data()

# --- 2. مدل‌سازی هوشمند (Machine Learning) ---
# آموزش مدل برای محاسبه اهمیت ویژگی‌ها
X = df[['Commitment', 'Justice', 'LMX_Manager_Rel', 'Burnout', 'Job_Fit', 'Tenure_Years']]
y = df['Will_Leave']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- 3. داشبورد ---

st.title("🧩 داشبورد هوشمند نگهداشت سرمایه انسانی")
st.markdown("تحلیل شاخص‌های روانشناسی صنعتی برای پیش‌گیری از خروج نخبگان")

# --- سایدبار و فیلترها ---
st.sidebar.header("فیلترهای نمایش")
dept_filter = st.sidebar.multiselect(
    "انتخاب دپارتمان:",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

filtered_df = df[df['Department'].isin(dept_filter)]

# --- بخش اول: نمای کلی و وضعیت اضطراری ---
col1, col2, col3, col4 = st.columns(4)

avg_risk = filtered_df['Risk_Score'].mean()
high_risk_count = filtered_df[filtered_df['Risk_Score'] > 75].shape[0]
avg_burnout = filtered_df['Burnout'].mean()
avg_justice = filtered_df['Justice'].mean()

col4.metric("میانگین ریسک خروج سازمان", f"{avg_risk:.1f}%", delta_color="inverse", delta=f"{avg_risk-50:.1f}")
col3.metric("تعداد کارکنان در منطقه قرمز", f"{high_risk_count} نفر", delta_color="inverse", delta="خطرناک")
col2.metric("میانگین فرسودگی شغلی", f"{avg_burnout:.1f} / 10", delta_color="inverse", delta=f"{avg_burnout-5:.1f}")
col1.metric("ادراک عدالت سازمانی", f"{avg_justice:.1f} / 10", delta=f"{avg_justice-5:.1f}")

st.markdown("---")

# --- بخش دوم: تحلیل ریشه‌ای (چرا افراد می‌روند؟) ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 عوامل اصلی روانشناختی مؤثر بر خروج")
    # استخراج اهمیت ویژگی‌ها از مدل
    feature_importance = pd.DataFrame({
        'Feature': ['تعهد عاطفی', 'عدالت سازمانی', 'رابطه با مدیر (LMX)', 'فرسودگی شغلی', 'تناسب شغل', 'سابقه کار'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', 
                     title="وزن هر شاخص در تصمیم به خروج (بر اساس هوش مصنوعی)",
                     color='Importance', color_continuous_scale='Redor')
    st.plotly_chart(fig_imp, use_container_width=True)

with c2:
    st.subheader("رادار سلامت روان تیم‌ها")
    # آماده‌سازی داده برای نمودار رادار
    radar_data = filtered_df.groupby('Department')[['Commitment', 'Justice', 'LMX_Manager_Rel', 'Job_Fit']].mean().reset_index()
    # نرمالایز کردن معکوس برای فرسودگی (چون کمش خوبه)
    radar_data['Burnout_Reverse'] = 10 - filtered_df.groupby('Department')['Burnout'].mean().values
    
    categories = ['تعهد', 'عدالت', 'رابطه با مدیر', 'تناسب شغل', 'عدم فرسودگی']
    
    fig_radar = go.Figure()
    
    for i, row in radar_data.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Commitment'], row['Justice'], row['LMX_Manager_Rel'], row['Job_Fit'], row['Burnout_Reverse']],
            theta=categories,
            fill='toself',
            name=row['Department']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="مقایسه دپارتمان‌ها"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- بخش سوم: لیست هشدار (Action List) ---
st.subheader("🚨 لیست هشدار: کارکنان با احتمال خروج بالا")
st.info("افراد زیر بر اساس ترکیب نمرات فرسودگی بالا، عدالت پایین و تعهد کم شناسایی شده‌اند.")

high_risk_employees = filtered_df[filtered_df['Risk_Score'] > 70].sort_values('Risk_Score', ascending=False)

# اضافه کردن ستون "دلیل اصلی" به صورت متنی برای نمایش
def identify_reason(row):
    reasons = []
    if row['Burnout'] > 7: reasons.append("فرسودگی شدید")
    if row['Justice'] < 4: reasons.append("احساس بی‌عدالتی")
    if row['LMX_Manager_Rel'] < 4: reasons.append("رابطه بد با مدیر")
    if row['Commitment'] < 4: reasons.append("عدم تعهد")
    return "، ".join(reasons) if reasons else "ریسک ترکیبی"

high_risk_employees['Main_Risk_Factor'] = high_risk_employees.apply(identify_reason, axis=1)

st.dataframe(
    high_risk_employees[['ID', 'Department', 'Risk_Score', 'Main_Risk_Factor', 'Burnout', 'Justice', 'LMX_Manager_Rel']].style.background_gradient(subset=['Risk_Score'], cmap='Reds'),
    use_container_width=True
)

# --- بخش چهارم: شبیه‌ساز پیشگیری (What-If Analysis) ---
st.markdown("---")
st.subheader("🛠️ شبیه‌ساز تصمیم‌گیری مدیریتی")
st.markdown("اگر وضعیت شاخص‌های روانشناسی را بهبود دهید، نرخ ریزش چقدر کاهش می‌یابد؟")

col_sim1, col_sim2, col_sim3 = st.columns(3)

with col_sim1:
    improve_justice = st.slider("افزایش احساس عدالت (%)", 0, 50, 0)
with col_sim2:
    reduce_burnout = st.slider("کاهش فرسودگی شغلی (%)", 0, 50, 0)
with col_sim3:
    improve_lmx = st.slider("بهبود رابطه با مدیران (%)", 0, 50, 0)

# محاسبه تاثیر شبیه‌سازی
current_high_risk_count = len(filtered_df[filtered_df['Risk_Score'] > 60])

# کپی دیتافریم برای شبیه‌سازی
sim_df = filtered_df.copy()

# اعمال تغییرات
sim_df['Justice'] = sim_df['Justice'] * (1 + improve_justice/100)
sim_df['Burnout'] = sim_df['Burnout'] * (1 - reduce_burnout/100)
sim_df['LMX_Manager_Rel'] = sim_df['LMX_Manager_Rel'] * (1 + improve_lmx/100)

# محاسبه مجدد ریسک
new_risk_score = (
    (sim_df['Burnout'] * 1.5) + 
    ((11 - sim_df['Justice']) * 1.2) + 
    ((11 - sim_df['Commitment']) * 1.0) +
    ((11 - sim_df['LMX_Manager_Rel']) * 0.8)
)
sim_df['New_Risk'] = ((new_risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())) * 100

new_high_risk_count = len(sim_df[sim_df['New_Risk'] > 60])
saved_employees = current_high_risk_count - new_high_risk_count

st.success(f"🎉 با اعمال این تغییرات، شما می‌توانید از خروج تقریبی **{saved_employees} نفر** جلوگیری کنید!")

# نمودار مقایسه قبل و بعد
fig_sim = go.Figure(data=[
    go.Bar(name='وضعیت فعلی', x=['کارکنان در معرض ریسک'], y=[current_high_risk_count], marker_color='indianred'),
    go.Bar(name='بعد از بهبود', x=['کارکنان در معرض ریسک'], y=[new_high_risk_count], marker_color='lightgreen')
])
fig_sim.update_layout(title="تأثیر مداخلات روانشناسی بر حفظ نیروی انسانی")
st.plotly_chart(fig_sim, use_container_width=True)
