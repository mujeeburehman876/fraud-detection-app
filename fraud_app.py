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
# 1. PAGE CONFIGURATION (MUST BE FIRST & ONLY ONCE)
# ==========================================
st.set_page_config(
    page_title="🛡️ Fraud Detection System", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 0. LIBRARY CHECK
# ==========================================
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# ==========================================
# 2. UI SETUP
# ==========================================
# DELETE THE SECOND st.set_page_config HERE

st.title("🛡️ Credit Card Fraud Detection Dashboard")
st.markdown("""
<style>
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #007bff;
    }
    .metric-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 5px;
    }
</style>
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
# 3. SIDEBAR CONTROLS (Global Settings)
# ==========================================
st.sidebar.header("⚙️ Global Settings")

# Data Upload Override
uploaded_file = st.sidebar.file_uploader("Upload CSV (Optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source = "Uploaded CSV"
else:
    df = df_original.copy()

random_seed = int(st.sidebar.number_input('Random Seed', value=42, step=1))

st.sidebar.info(f"📁 Data Source: {source}")
st.sidebar.info(f"📊 Total Transactions: {len(df)}")

# ==========================================
# 4. TABS SETUP
# ==========================================
tab_train, tab_test = st.tabs(["🏋️ 1. Training Configuration", "🧪 2. Testing & Evaluation"])

# ==========================================
# TAB 1: TRAINING LOGIC
# ==========================================
with tab_train:
    st.subheader("🔧 Configure Training Environment")
    
    col_control1, col_control2 = st.columns(2)
    
    with col_control1:
        st.markdown("### A. Data Split")
        # Slider controls TRAINING size (Default 80%)
        train_split_percent = st.slider("Select Training Data Percentage", 50, 95, 80)
        test_split_percent = 100 - train_split_percent
        
        # Convert to decimal for sklearn
        test_size_decimal = test_split_percent / 100.0
        
        st.caption(f"**Training Set:** {train_split_percent}% | **Test Set:** {test_split_percent}%")

    with col_control2:
        st.markdown("### B. Sampling Technique")
        st.caption("Applied to **Training Data Only** to fix imbalance.")
        
        sampling_options = ["Normal (No Sampling)", "SMOTE", "Random Over-Sampling", "Random Under-Sampling"]
        selected_sampling = st.radio("Select Method:", sampling_options)

        if not IMBLEARN_AVAILABLE and selected_sampling != "Normal (No Sampling)":
            st.error("⚠️ `imbalanced-learn` is not installed. Please install it.")

    # --- PROCESSING LOGIC ---
    # 1. Cleaning
    if 'Time' in df.columns:
        df = df.drop(['Time'], axis=1)
    df.dropna(inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    # 2. Splitting (Stratified to keep fraud ratio consistent)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_decimal, random_state=random_seed, stratify=y
    )

    # 3. Scaling (Fit on Train, Transform on Test)
    scaler = StandardScaler()
    if 'Amount' in X_train.columns:
        X_train['Amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
        X_test['Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
    else:
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 4. Sampling (Apply ONLY to X_train/y_train)
    X_train_resampled, y_train_resampled = X_train, y_train # Default
    
    if IMBLEARN_AVAILABLE and selected_sampling != "Normal (No Sampling)":
        try:
            if selected_sampling == "SMOTE":
                sampler = SMOTE(random_state=random_seed)
            elif selected_sampling == "Random Over-Sampling":
                sampler = RandomOverSampler(random_state=random_seed)
            elif selected_sampling == "Random Under-Sampling":
                sampler = RandomUnderSampler(random_state=random_seed)
            
            # Apply Sampling
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            
        except Exception as e:
            st.error(f"Sampling Error: {e}")

    # --- DISPLAY STATS ---
    st.divider()
    st.markdown("### 📊 Data Stats (Live Update)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", len(df))
    c2.metric(f"Training Set ({train_split_percent}%)", 
              f"{len(X_train)} rows", 
              delta=f"Became {len(X_train_resampled)} after {selected_sampling}" if selected_sampling != "Normal (No Sampling)" else None)
    c3.metric(f"Test Set ({test_split_percent}%)", f"{len(X_test)} rows", "Reserved for Tab 2")
    
    st.info("✅ Configuration Ready. The model will define patterns based on the data above. Switch to Tab 2 to test it.")

# ==========================================
# MODEL TRAINING (HAPPENS AUTOMATICALLY)
# ==========================================
# We train the model here so variables are available for Tab 2
model = RandomForestClassifier(n_estimators=50, random_state=random_seed)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)

# ==========================================
# TAB 2: TESTING LOGIC
# ==========================================
with tab_test:
    st.subheader(f"🧪 Testing Results (On {test_split_percent}% Reserved Data)")
    st.caption(f"The model is now being tested on **{len(X_test)}** unseen transactions.")

    # 1. Calculate Metrics
    cm = confusion_matrix(y_test, y_pred)
    # Handle single-class edge cases
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0

    # 2. Display Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.2%}", help="Overall correctness")
    m2.metric("Precision", f"{prec:.2%}", help="How many detected frauds were actually fraud?")
    m3.metric("Recall", f"{rec:.2%}", help="How many actual frauds did we catch? (Critical)")
    m4.metric("F1-Score", f"{f1:.2%}", help="Balance of Precision/Recall")
    m5.metric("Specificity", f"{specificity:.2%}", help="True Negatives")

    st.divider()

    # 3. Visualizations
    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        st.markdown("### 📉 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Pred Normal', 'Pred Fraud'],
                    yticklabels=['Actual Normal', 'Actual Fraud'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig_cm)
        st.info(f"Total Tested: {len(X_test)}. \nTop-Left: Correct Normal. Bottom-Right: Correct Fraud.")

    with col_viz2:
        st.markdown("### 🧠 Feature Importance")
        importances = model.feature_importances_
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        fig_feat, ax_feat = plt.subplots()
        plt.title("Top 10 Features Driving Decisions")
        plt.bar(range(len(indices)), importances[indices], align="center", color="#4CAF50")
        plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=45)
        plt.xlim([-1, len(indices)])
        st.pyplot(fig_feat)