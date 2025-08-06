import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# إعداد الصفحة
st.set_page_config(page_title="HR Dashboard", layout="wide")

# تحميل البيانات والنموذج (من نفس المجلد)
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
model = joblib.load("model.pkl")

# 🔧 المعالجة الأولية للمدخلات
def preprocess_input(user_input):
    cat_cols = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
    label_encoders = {col: LabelEncoder().fit(data[col]) for col in cat_cols}

    for col in cat_cols:
        user_input[col] = label_encoders[col].transform([user_input[col]])[0]

    input_df = pd.DataFrame([user_input])
    input_df = input_df[[
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]]

    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)
    return input_scaled

# 🎛️ الفلاتر الجانبية
st.sidebar.title("🎛️ Filtres")
age_range = st.sidebar.slider("Âge", int(data["Age"].min()), int(data["Age"].max()), (30, 40))
gender_filter = st.sidebar.selectbox("Genre", ["Tous"] + list(data["Gender"].unique()))
jobrole_filter = st.sidebar.selectbox("Poste", ["Tous"] + list(data["JobRole"].unique()))

# تطبيق الفلاتر
filtered_data = data[(data["Age"] >= age_range[0]) & (data["Age"] <= age_range[1])]
if gender_filter != "Tous":
    filtered_data = filtered_data[filtered_data["Gender"] == gender_filter]
if jobrole_filter != "Tous":
    filtered_data = filtered_data[filtered_data["JobRole"] == jobrole_filter]

# العنوان الرئيسي
st.title(" Tableau de bord RH - Analyse des employés")

# 👤 الإحصائيات
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Employés", len(filtered_data))
with col2:
    attrition_rate = (filtered_data["Attrition"] == "Yes").mean() * 100
    st.metric("Taux de départ", f"{attrition_rate:.1f}%")
with col3:
    avg_income = filtered_data["MonthlyIncome"].mean()
    st.metric("Salaire moyen", f"{avg_income:.0f} DH")

st.markdown("---")

# 📊 الرسوم
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔻 Répartition des départs")
    st.bar_chart(filtered_data["Attrition"].value_counts())

with col2:
    st.subheader("🔁 OverTime vs Attrition")
    overtime_df = pd.crosstab(filtered_data["OverTime"], filtered_data["Attrition"])
    st.bar_chart(overtime_df)

st.subheader("📈 Job Satisfaction vs Attrition")
fig1, ax1 = plt.subplots()
sns.boxplot(data=filtered_data, x="Attrition", y="JobSatisfaction", ax=ax1)
st.pyplot(fig1)

st.subheader("💰 Distribution des salaires")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_data["MonthlyIncome"], bins=20, kde=True, ax=ax2)
st.pyplot(fig2)

# 🔮 نموذج التنبؤ
st.sidebar.title("🔮 Prédiction départ employé")
user_input = {
    "Age": st.sidebar.slider("Âge", 18, 60, 30),
    "BusinessTravel": st.sidebar.selectbox("Déplacement", data["BusinessTravel"].unique()),
    "DailyRate": st.sidebar.slider("Daily Rate", 100, 1500, 800),
    "Department": st.sidebar.selectbox("Département", data["Department"].unique()),
    "DistanceFromHome": st.sidebar.slider("Distance", 1, 30, 10),
    "Education": st.sidebar.slider("Éducation", 1, 5, 3),
    "EducationField": st.sidebar.selectbox("Domaine", data["EducationField"].unique()),
    "EnvironmentSatisfaction": st.sidebar.slider("Satisfaction environnement", 1, 4, 3),
    "Gender": st.sidebar.selectbox("Genre", data["Gender"].unique()),
    "HourlyRate": st.sidebar.slider("Taux horaire", 30, 100, 60),
    "JobInvolvement": st.sidebar.slider("Engagement", 1, 4, 3),
    "JobLevel": st.sidebar.slider("Niveau d'emploi", 1, 5, 2),
    "JobRole": st.sidebar.selectbox("Poste", data["JobRole"].unique()),
    "JobSatisfaction": st.sidebar.slider("Satisfaction", 1, 4, 3),
    "MaritalStatus": st.sidebar.selectbox("État civil", data["MaritalStatus"].unique()),
    "MonthlyIncome": st.sidebar.slider("Salaire mensuel", 1000, 20000, 5000),
    "MonthlyRate": st.sidebar.slider("Taux mensuel", 2000, 25000, 10000),
    "NumCompaniesWorked": st.sidebar.slider("Expérience entreprises", 0, 10, 2),
    "OverTime": st.sidebar.selectbox("Heures sup", data["OverTime"].unique()),
    "PercentSalaryHike": st.sidebar.slider("Augmentation", 10, 25, 15),
    "PerformanceRating": st.sidebar.slider("Évaluation performance", 1, 4, 3),
    "RelationshipSatisfaction": st.sidebar.slider("Relations", 1, 4, 3),
    "StockOptionLevel": st.sidebar.slider("Actions", 0, 3, 1),
    "TotalWorkingYears": st.sidebar.slider("Années de travail", 0, 40, 10),
    "TrainingTimesLastYear": st.sidebar.slider("Formations", 0, 6, 2),
    "WorkLifeBalance": st.sidebar.slider("Équilibre vie-travail", 1, 4, 3),
    "YearsAtCompany": st.sidebar.slider("Années dans l'entreprise", 0, 40, 5),
    "YearsInCurrentRole": st.sidebar.slider("Ancienneté dans poste", 0, 20, 3),
    "YearsSinceLastPromotion": st.sidebar.slider("Depuis promotion", 0, 15, 2),
    "YearsWithCurrManager": st.sidebar.slider("Avec manager actuel", 0, 17, 3)
}

if st.sidebar.button("Prédire"):
    input_processed = preprocess_input(user_input)
    prediction = model.predict(input_processed)[0]
    if prediction == 1:
        st.sidebar.error("⚠️ Risque élevé de départ.")
    else:
        st.sidebar.success("✅ Employé stable.")
