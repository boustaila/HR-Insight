# 💼 HR-Insight — Tableau de bord RH avec prédiction d'attrition

Un projet complet de **Data Science** et **Business Intelligence RH** combinant :

- Visualisation interactive des données RH
- Prédiction du départ des employés via Machine Learning (XGBoost)
- Interface utilisateur avec **Streamlit**

---

## 🎯 Objectif

L’objectif est d’offrir un outil interactif permettant :

✅ D’analyser les données RH (satisfaction, revenu, rôle, etc.)  
✅ De détecter les tendances menant au départ d’un collaborateur  
✅ De prédire si un employé donné est à risque de quitter l’entreprise

---

## 📊 Aperçu du dashboard

![Aperçu du dashboard](./assets/dashboard_sample.png) <!-- Ajoute une image plus tard si nécessaire -->

---

## 🧠 Modèle ML utilisé

- **XGBoostClassifier**
- Traitement des déséquilibres avec **SMOTE**
- Évaluation via Accuracy, F1-score, ROC-AUC
- Sauvegarde du modèle avec `joblib`

---

## 🔍 Filtres interactifs disponibles

- Âge
- Sexe
- Poste occupé
- + Graphiques dynamiques : taux de départ, salaire, satisfaction...

---

## 🚀 Lancer l’application localement

### 1. Cloner le repo

```bash
git clone https://github.com/boustaila/HR-Insight.git
cd HR-Insight

### Installer les dépendances
pip install -r requirements.txt
### Lancer le dashboard
streamlit run main.py
