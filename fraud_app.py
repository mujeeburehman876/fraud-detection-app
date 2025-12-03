import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. LIBRARY CHECK (Imbalanced-Learn)
# ==========================================
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

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
**Sampling techniques are applied only to the training data to ensure validity.**
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & HANDLING
# ==========================================
@st.cache_data
def load_data():
    """
    Loads creditcard.csv, creditcard.csv.zip, or generates synthetic data 
    if files are missing.
    """
    df = None
    data_source = "Synthetic (Demo)"

    # 1. Try loading raw CSV
    if os.path.exists('creditcard.csv'):
        df = pd.read_csv('creditcard.csv')
        data_source = "Local CSV"
    
    # 2. Try loading Zipped CSV (Useful for GitHub/Large files)
    elif os.path.exists('creditcard.csv.zip'):
        df = pd.read_csv('creditcard.csv.zip', compression='zip')
        data_source = "Local ZIP"
        
    elif os.path.exists('creditcard.zip'):
        df = pd.read_csv('creditcard.zip', compression='zip')
        data_source = "Local ZIP"

    # 3. Generate Synthetic Data if no file found
    if df is None:
        np.random.seed(42)
        n_rows = 10000
        # Generate 28 anonymized PCA features (V1-V28)
        data = np.random.randn(n_rows, 28) 
        df = pd.DataFrame(data, columns=[f'V{i}' for i in range(1, 29)])
        # Time and Amount
        df['Time'] = np.arange(n_rows)
        df['Amount'] = np.random.uniform(0, 500, n_rows)
        # Class (0 = Normal, 1 = Fraud) - Create imbalance (2% fraud)
        df['Class'] = np.random.choice([0, 1], size=n_rows, p=[0.98, 0.02])
    else:
        # Sampling for speed if dataset is huge (optional)
        if len(df) > 50000:
            df = df.sample(50000, random_state=42)
    
    return df, data_source

# Load Data
with st.spinner('Loading and analyzing data...'):
    df_original, source = load_data()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Model Parameters")

# Data Upload Override
uploaded_file = st.sidebar.file_uploader("Upload CSV (Optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source = "Uploaded CSV"
else:
    df = df_original.copy()

split_size = st.sidebar.slider('Test Set Size', 0.1, 0.5, 0.2)
random_seed = int(st.sidebar.number_input('Random Seed', value=42, step=1))

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Sampling Technique")
st.sidebar.caption("Applied to Training Data Only")

# SAMPLING SELECTION
sampling_options = ["Normal (No Sampling)", "SMOTE", "Random Over-Sampling", "Random Under-Sampling"]
selected_sampling = st.sidebar.radio("Select Method:", sampling_options)

st.sidebar.info(f"📁 Data Source: {source}")
st.sidebar.info(f"📊 Total Transactions: {len(df)}")

if not IMBLEARN_AVAILABLE and selected_sampling != "Normal (No Sampling)":
    st.sidebar.error("⚠️ `imbalanced-learn` is not installed. Please install it to use sampling methods.")

# ==========================================
# 4. EXPLORATORY DATA ANALYSIS (Quick View)
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Class Distribution")
    if 'Class' not in df.columns:
        st.error("Dataset must contain a 'Class' column.")
        st.stop()
        
    counts = df['Class'].value_counts()
    # Handle cases where a class might be missing
    normal_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    
    # Simple pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie([normal_count, fraud_count], labels=['Normal', 'Fraud'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    st.pyplot(fig_pie)
    st.caption(f"Normal: {normal_count} | Fraud: {fraud_count}")

with col2:
    st.subheader("2. Dataset Preview")
    st.dataframe(df.head(8), use_container_width=True)

# ==========================================
# 5. PREPROCESSING & SPLITTING
# ==========================================
st.divider()
st.subheader("3. Model Training & Evaluation")

# 1. Feature Scaling Setup
# We drop Time as it is usually irrelevant for this specific generic model
if 'Time' in df.columns:
    df = df.drop(['Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# 2. Train/Test Split
# We split BEFORE sampling to prevent data leakage into the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_size, random_state=random_seed, stratify=y
)

# 3. Scaling (StandardScaler)
# Fit on Train, Transform on Train and Test
scaler = StandardScaler()

# Check if Amount exists to scale it specifically, or scale all features
if 'Amount' in X_train.columns:
    # Scale 'Amount' column specifically
    X_train['Amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
    X_test['Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
else:
    # If generic data, scale everything
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ==========================================
# 6. APPLY SAMPLING (TRAIN SET ONLY)
# ==========================================
X_train_resampled, y_train_resampled = X_train, y_train  # Default: No change

if IMBLEARN_AVAILABLE:
    try:
        if selected_sampling == "SMOTE":
            smote = SMOTE(random_state=random_seed)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            st.toast("✅ SMOTE Applied successfully!")
            
        elif selected_sampling == "Random Over-Sampling":
            ros = RandomOverSampler(random_state=random_seed)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
            st.toast("✅ Random Over-Sampling Applied!")
            
        elif selected_sampling == "Random Under-Sampling":
            rus = RandomUnderSampler(random_state=random_seed)
            X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
            st.toast("✅ Random Under-Sampling Applied!")
            
    except Exception as e:
        st.error(f"Error applying sampling: {e}")
else:
    if selected_sampling != "Normal (No Sampling)":
        st.warning("Imbalanced-learn library not found. Proceeding with Normal data.")

# Show sampling impact
if selected_sampling != "Normal (No Sampling)":
    st.info(f"**Sampling Result:** Training data changed from {len(X_train)} to {len(X_train_resampled)} records.")

# ==========================================
# 7. MODEL TRAINING
# ==========================================
model = RandomForestClassifier(n_estimators=50, random_state=random_seed)

with st.spinner(f"Training RandomForest with {selected_sampling}..."):
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

# ==========================================
# 8. METRICS CALCULATION
# ==========================================
# Confusion Matrix components for Specificity
cm = confusion_matrix(y_test, y_pred)

# Safe unpack of confusion matrix (handles edge cases with 1 class)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    # Fallback if model only predicted one class
    tn, fp, fn, tp = 0, 0, 0, 0 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
f1 = f1_score(y_test, y_pred, zero_division=0)
specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0

# Display Metrics using columns
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{accuracy:.2%}", help="Overall correctness")
m2.metric("Precision", f"{precision:.2%}", help="How many detected frauds were actually fraud?")
m3.metric("Recall", f"{recall:.2%}", help="How many actual frauds did we catch?")
m4.metric("F1-Score", f"{f1:.2%}", help="Balance between Precision and Recall")
m5.metric("Specificity", f"{specificity:.2%}", help="Ability to identify normal transactions")

# ==========================================
# 9. VISUALIZATION
# ==========================================
c_chart1, c_chart2 = st.columns(2)

with c_chart1:
    st.markdown("### 📉 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Pred Normal', 'Pred Fraud'],
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
    plt.bar(range(len(indices)), importances[indices], align="center", color="#4CAF50")
    plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=45)
    plt.xlim([-1, len(indices)])
    st.pyplot(fig_feat)

# ==========================================
# 10. SUMMARY
# ==========================================
st.success(f"""
**Analysis Complete:** - Technique Used: **{selected_sampling}**
- The model successfully processed **{len(X_test)}** test transactions. 
- It identified **{tp}** fraudulent transactions correctly out of **{tp+fn}** actual frauds.
""")
