import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="🛡️ Fraud Detection System", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ Credit Card Fraud Detection Dashboard")
st.markdown("""
<style>
    .metric-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 5px;
    }
</style>
This application analyzes transaction data to identify fraudulent activities using Machine Learning.
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & HANDLING
# ==========================================
@st.cache_data
def load_data():
    """Loads creditcard.csv or generates synthetic data if file missing."""
    try:
        # Attempt to load local file
        df = pd.read_csv('creditcard.csv')
        data_source = "Local CSV"
        # Sampling for speed if dataset is huge (optional)
        if len(df) > 50000:
            df = df.sample(50000, random_state=42)
    except FileNotFoundError:
        # Generate synthetic data for demonstration
        data_source = "Synthetic (Demo)"
        np.random.seed(42)
        n_rows = 5000
        # Generate 28 anonymized PCA features (V1-V28)
        data = np.random.randn(n_rows, 28) 
        df = pd.DataFrame(data, columns=[f'V{i}' for i in range(1, 29)])
        # Time and Amount
        df['Time'] = np.arange(n_rows)
        df['Amount'] = np.random.uniform(0, 500, n_rows)
        # Class (0 = Normal, 1 = Fraud) - Create imbalance (2% fraud)
        df['Class'] = np.random.choice([0, 1], size=n_rows, p=[0.98, 0.02])
    
    return df, data_source

# Load Data
with st.spinner('Loading and analyzing data...'):
    df, source = load_data()

# Sidebar Controls
st.sidebar.header("⚙️ Model Parameters")
split_size = st.sidebar.slider('Test Set Size', 0.1, 0.5, 0.2)
random_seed = st.sidebar.number_input('Random Seed', 42)
st.sidebar.info(f"📁 Data Source: {source}")
st.sidebar.info(f"📊 Total Transactions: {len(df)}")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (Quick View)
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Class Distribution")
    counts = df['Class'].value_counts()
    fraud_count = counts.get(1, 0)
    normal_count = counts.get(0, 0)
    
    # Simple pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie([normal_count, fraud_count], labels=['Normal', 'Fraud'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    st.pyplot(fig_pie)
    st.caption(f"Normal: {normal_count} | Fraud: {fraud_count}")

with col2:
    st.subheader("2. Dataset Preview")
    st.dataframe(df.head(8), use_container_width=True)
    st.markdown(f"**Insight:** The dataset is highly unbalanced. Standard accuracy is not enough; we need Precision and Recall.")

# ==========================================
# 4. PREPROCESSING & TRAINING
# ==========================================
st.divider()
st.subheader("3. Model Training & Evaluation")

# Feature Scaling (Amount is often unscaled in raw data)
sc = StandardScaler()
df['Amount_Scaled'] = sc.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=random_seed, stratify=y)

# Model Training
model = RandomForestClassifier(n_estimators=50, random_state=random_seed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================
# 5. METRICS CALCULATION
# ==========================================
# Confusion Matrix components for Specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred) # Sensitivity
f1 = f1_score(y_test, y_pred)
specificity = tn / (tn + fp) # Simplicity / True Negative Rate

# Display Metrics using columns
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{accuracy:.2%}", help="Overall correctness")
m2.metric("Precision", f"{precision:.2%}", help="How many detected frauds were actually fraud?")
m3.metric("Recall", f"{recall:.2%}", help="How many actual frauds did we catch?")
m4.metric("F1-Score", f"{f1:.2%}", help="Balance between Precision and Recall")
m5.metric("Specificity", f"{specificity:.2%}", help="Ability to identify normal transactions")

# ==========================================
# 6. VISUALIZATION
# ==========================================
c_chart1, c_chart2 = st.columns(2)

with c_chart1:
    st.markdown("### 📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Normal', 'Predicted Fraud'],
                yticklabels=['Actual Normal', 'Actual Fraud'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig_cm)
    st.info("Top-Left: Correctly cleared. Bottom-Right: Fraud caught.")

with c_chart2:
    st.markdown("### 🧠 Feature Importance")
    importances = model.feature_importances_
    # Get top 10 features
    indices = np.argsort(importances)[::-1][:10]
    
    fig_feat, ax_feat = plt.subplots()
    plt.title("Top 10 Features Driving Fraud Detection")
    plt.bar(range(10), importances[indices], align="center", color="#4CAF50")
    plt.xticks(range(10), [X.columns[i] for i in indices], rotation=45)
    plt.xlim([-1, 10])
    st.pyplot(fig_feat)

# ==========================================
# 7. SUMMARY
# ==========================================
st.success(f"""
**Analysis Complete:** The model successfully processed **{len(X_test)}** test transactions. 
It identified **{tp}** fraudulent transactions correctly out of **{tp+fn}** actual frauds.
""")