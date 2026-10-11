"""
Tata Motors EV Marketing Analysis - Interactive Web Application
Built with Streamlit for MCA Semester-End Capstone Project
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure parent and scripts paths are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# Import ML engine classes for deserialization compatibility
from scripts.ml_engine import (
    DataPreprocessor, LinearRegressionModel, LogisticRegressionModel,
    RandomForestRegressorModel, RandomForestClassifierModel,
    GradientBoostingRegressorModel, GradientBoostingClassifierModel,
    KMeansClusteringModel
)

# Page Configuration
st.set_page_config(
    page_title="Tata Motors EV Marketing Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-left: 5px solid #2B6CB0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Data Caching
@st.cache_data
def load_data():
    data_path = os.path.join(PROJECT_ROOT, "data", "tata_ev_marketing_enriched.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "data", "tata_ev_marketing.csv")
    return pd.read_csv(data_path)

@st.cache_resource
def load_models():
    sales_path = os.path.join(PROJECT_ROOT, "models", "sales_model.pkl")
    conv_path = os.path.join(PROJECT_ROOT, "models", "conversion_model.pkl")
    sales_bundle = joblib.load(sales_path) if os.path.exists(sales_path) else None
    conv_bundle = joblib.load(conv_path) if os.path.exists(conv_path) else None
    return sales_bundle, conv_bundle

df = load_data()
sales_bundle, conv_bundle = load_models()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/8e/Tata_logo.svg", width=120)
st.sidebar.title("⚡ Navigation")
app_mode = st.sidebar.radio(
    "Select Module:",
    [
        "Executive Dashboard",
        "Marketing Analytics",
        "EV Portfolio Insights",
        "Customer Demographics",
        "Predictive ML Simulator",
        "Statistical Testing & Metrics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project Title:**
Tata Motors EV Marketing Analysis

**Program:**
MCA - Semester-End Capstone
**Specialization:**
Advanced Data Science & ML
""")

# ==============================================================================
# 1. EXECUTIVE DASHBOARD
# ==============================================================================
if app_mode == "Executive Dashboard":
    st.markdown('<div class="main-header">⚡ Tata Motors EV Marketing Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Data Science & Machine Learning Capstone Analytics Portal</div>', unsafe_allow_html=True)
    
    # Top KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total_leads = len(df)
    conv_rate = (df['Lead_Status'] == 'Converted').mean() * 100
    total_revenue_cr = df['Sales_Amount'].sum() / 1e7
    avg_price_lakh = df['Purchase_Price'].mean() / 1e5
    test_drive_pct = (df['Test_Drive'] == 'Yes').mean() * 100
    
    col1.metric("Total EV Inquiries", f"{total_leads:,}")
    col2.metric("Conversion Rate", f"{conv_rate:.1f}%")
    col3.metric("Total Revenue", f"₹ {total_revenue_cr:.2f} Cr")
    col4.metric("Avg EV Price", f"₹ {avg_price_lakh:.1f} Lakh")
    col5.metric("Test Drive Rate", f"{test_drive_pct:.1f}%")
    
    st.markdown("---")
    
    c_left, c_right = st.columns([1.2, 1])
    
    with c_left:
        st.subheader("🎯 Capstone Project Objective & Overview")
        st.write("""
        This project analyzes the customer acquisition funnel and marketing touchpoints for 
        **Tata Motors Passenger Electric Vehicles (EV)** across major metropolitan and Tier-2 automotive hubs in India.
        
        **Core Methodological Pillars:**
        - **Descriptive & Bivariate EDA:** Identifying demographic clusters, model preferences, and high-yield marketing channels.
        - **Inferential Statistical Testing:** Evaluating hypothesis significance (Chi-Square test, Welch's t-test, ANOVA).
        - **Dual-Track Machine Learning:**
            - **Sales Revenue Prediction (Regression):** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
            - **Customer Conversion Propensity (Classification):** Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier.
        - **Customer Segmentation:** Unsupervised K-Means clustering isolating 4 distinct behavioral buyer personas.
        """)
        
        st.markdown("""
        > **Academic Notice:**
        > *This analysis utilizes a statistically calibrated academic synthetic dataset modeled after the Indian passenger EV market. It does not disclose proprietary or confidential Tata Motors customer records.*
        """)

    with c_right:
        st.subheader("📊 Lead Conversion Funnel Breakdown")
        status_counts = df['Lead_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = ['#2E7D32', '#C62828', '#1565C0', '#F57C00']
        ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=colors, startangle=140, explode=[0.05, 0, 0, 0])
        ax.set_title("Current Lead Distribution Status", fontweight='bold')
        st.pyplot(fig)

# ==============================================================================
# 2. MARKETING ANALYTICS
# ==============================================================================
elif app_mode == "Marketing Analytics":
    st.header("📈 Marketing Channel & Campaign Performance")
    
    tab1, tab2 = st.tabs(["Channel Attribution", "Campaign Performance & Engagement"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Channel Lead Volume vs Conversion Yield")
            ch_summary = df.groupby('Marketing_Channel').agg(
                Leads=('Customer_ID', 'count'),
                Converted=('Conversion_Flag', 'sum')
            )
            ch_summary['Conv_Rate'] = (ch_summary['Converted'] / ch_summary['Leads']) * 100
            ch_summary = ch_summary.sort_values(by='Conv_Rate', ascending=False)
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.barplot(x=ch_summary['Conv_Rate'], y=ch_summary.index, palette='Blues_r', ax=ax)
            ax.set_xlabel("Conversion Rate (%)")
            ax.set_ylabel("Marketing Channel")
            for p in ax.patches:
                ax.annotate(f"{p.get_width():.1f}%", (p.get_width() + 0.5, p.get_y() + p.get_height() / 2), va='center')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Impact of Test Drive Completion")
            td_cross = pd.crosstab(df['Test_Drive'], df['Lead_Status'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(7, 4.5))
            td_cross[['Converted', 'Contacted', 'New', 'Lost']].plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
            ax.set_ylabel("Lead Ratio (%)")
            ax.set_xlabel("Completed Test Drive?")
            plt.xticks(rotation=0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
        st.info("💡 **Marketing Insight:** Dealership Test Drives elevate lead conversion from ~21% to over 68%, making experiential showroom drives the single most influential conversion accelerator.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Campaign Type Revenue Generation")
            camp_rev = df.groupby('Campaign_Type')['Sales_Amount'].sum() / 1e7
            fig, ax = plt.subplots(figsize=(7, 4.5))
            camp_rev.sort_values().plot(kind='barh', color='#2B6CB0', ax=ax)
            ax.set_xlabel("Realized Revenue (₹ Crores)")
            ax.set_ylabel("Campaign Type")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Digital Engagement: Website Visits vs Ad Exposure")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(data=df, x='Ad_Exposure', y='Website_Visits', hue='Conversion_Flag', palette='coolwarm', alpha=0.7, ax=ax)
            ax.set_xlabel("Advertisement Exposures")
            ax.set_ylabel("Website Visits")
            st.pyplot(fig)

# ==============================================================================
# 3. EV PORTFOLIO INSIGHTS
# ==============================================================================
elif app_mode == "EV Portfolio Insights":
    st.header("🚗 Tata Motors EV Portfolio Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inquiry Volume by Tata EV Model")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=df, x='EV_Model', order=['Punch EV', 'Nexon EV', 'Tiago EV', 'Curvv EV', 'Tigor EV', 'Harrier EV'], palette='viridis', ax=ax)
        ax.set_xlabel("Model Name")
        ax.set_ylabel("Inquiry Volume")
        plt.xticks(rotation=25)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Model-Wise Realized Revenue (₹ Crores)")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        rev_mod = (df.groupby('EV_Model')['Sales_Amount'].sum() / 1e7).reindex(['Punch EV', 'Nexon EV', 'Tiago EV', 'Curvv EV', 'Tigor EV', 'Harrier EV'])
        sns.barplot(x=rev_mod.index, y=rev_mod.values, palette='magma', ax=ax)
        ax.set_xlabel("EV Model")
        ax.set_ylabel("Sales Realized (₹ Cr)")
        plt.xticks(rotation=25)
        st.pyplot(fig)
        
    st.subheader("Portfolio Price Spectrum & Conversion Efficiency")
    mod_summary = df.groupby('EV_Model').agg(
        Total_Leads=('Customer_ID', 'count'),
        Avg_Price_INR=('Purchase_Price', 'mean'),
        Converted=('Conversion_Flag', 'sum'),
        Total_Revenue=('Sales_Amount', 'sum')
    )
    mod_summary['Conversion_Rate_%'] = (mod_summary['Converted'] / mod_summary['Total_Leads']) * 100
    mod_summary['Avg_Price_Lakh'] = mod_summary['Avg_Price_INR'] / 1e5
    mod_summary['Revenue_Cr'] = mod_summary['Total_Revenue'] / 1e7
    
    st.dataframe(
        mod_summary[['Total_Leads', 'Avg_Price_Lakh', 'Conversion_Rate_%', 'Revenue_Cr']].round(2),
        use_container_width=True
    )

# ==============================================================================
# 4. CUSTOMER DEMOGRAPHICS
# ==============================================================================
elif app_mode == "Customer Demographics":
    st.header("👥 Customer Demographics & Persona Segmentation")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Age'], bins=15, kde=True, color='#0288D1', ax=ax)
        ax.set_xlabel("Customer Age")
        st.pyplot(fig)
        
    with c2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        g_counts = df['Gender'].value_counts()
        ax.pie(g_counts, labels=g_counts.index, autopct='%1.1f%%', colors=['#1976D2', '#E91E63', '#9C27B0'])
        st.pyplot(fig)
        
    with c3:
        st.subheader("Top Geographic Hubs")
        fig, ax = plt.subplots(figsize=(6, 4))
        df['City'].value_counts().head(6).plot(kind='bar', color='#43A047', ax=ax)
        ax.set_ylabel("Inquiries")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("🤖 K-Means Unsupervised Customer Clusters")
    
    cluster_file = os.path.join(PROJECT_ROOT, "outputs", "results", "customer_clusters.csv")
    if os.path.exists(cluster_file):
        cdf = pd.read_csv(cluster_file)
        cdf['Avg_Price_Lakh'] = cdf['Avg_Price'] / 1e5
        cdf['Conversion_Rate_%'] = cdf['Conversion_Rate'] * 100
        st.dataframe(
            cdf[['Cluster', 'Persona_Name', 'Avg_Age', 'Avg_Price_Lakh', 'Avg_Web_Visits', 'Avg_Ad_Exposure', 'Conversion_Rate_%', 'Headcount']].round(2),
            use_container_width=True
        )

# ==============================================================================
# 5. PREDICTIVE ML SIMULATOR
# ==============================================================================
elif app_mode == "Predictive ML Simulator":
    st.header("🔮 Interactive Predictive Machine Learning Simulator")
    st.write("Input prospective customer parameters below to simulate **Conversion Propensity** and expected **Sales Revenue**.")
    
    if sales_bundle is None or conv_bundle is None:
        st.error("Model bundles not found! Please run the pipeline script first.")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                input_age = st.slider("Customer Age", min_value=21, max_value=68, value=35)
                input_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                input_city = st.selectbox("City", ['Mumbai', 'Delhi', 'Bengaluru', 'Pune', 'Hyderabad', 'Chennai', 'Ahmedabad', 'Kolkata', 'Jaipur', 'Kochi'])
                
            with col2:
                input_model = st.selectbox("Tata EV Model", ['Punch EV', 'Nexon EV', 'Tiago EV', 'Curvv EV', 'Tigor EV', 'Harrier EV'])
                default_prices = {
                    'Tiago EV': 950000, 'Punch EV': 1250000, 'Tigor EV': 1300000,
                    'Nexon EV': 1650000, 'Curvv EV': 1950000, 'Harrier EV': 2350000
                }
                input_price = st.number_input("Purchase Price (₹ INR)", value=default_prices[input_model], step=25000)
                input_channel = st.selectbox("Marketing Channel", ['Social Media', 'Google Search', 'TV Commercial', 'Dealership Visit', 'Print Media', 'Email Campaign'])
                
            with col3:
                input_ad_exp = st.slider("Ad Exposure Count", min_value=1, max_value=25, value=6)
                input_web_vis = st.slider("Website Visits", min_value=1, max_value=30, value=8)
                input_camp = st.selectbox("Campaign Type", ['Festive Offer', 'Green Mobility Drive', 'Exchange Bonus', 'Referral Program', 'New Launch Blitz'])
                input_td = st.radio("Test Drive Taken?", ["Yes", "No"], horizontal=True)
                input_interest = st.selectbox("Customer Interest Level", ["Low", "Medium", "High"], index=1)
                
            submit_btn = st.form_submit_button("⚡ Run Predictive Inference")
            
        if submit_btn:
            input_df = pd.DataFrame([{
                'Age': input_age,
                'Purchase_Price': input_price,
                'Ad_Exposure': input_ad_exp,
                'Website_Visits': input_web_vis,
                'Gender': input_gender,
                'City': input_city,
                'EV_Model': input_model,
                'Marketing_Channel': input_channel,
                'Campaign_Type': input_camp,
                'Test_Drive': input_td,
                'Customer_Interest': input_interest
            }])
            
            # Predict Conversion Probability
            prep_conv = conv_bundle['preprocessor']
            X_conv = prep_conv.transform(input_df)
            conv_prob = conv_bundle['model'].predict_proba(X_conv)[0, 1]
            conv_flag = int(conv_prob >= 0.5)
            
            # Predict Sales Amount
            prep_sales = sales_bundle['preprocessor']
            X_sales = prep_sales.transform(input_df)
            pred_sales = float(sales_bundle['model'].predict(X_sales)[0])
            pred_sales = max(0.0, pred_sales)
            
            st.markdown("---")
            st.subheader("📊 Model Inference Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Conversion Probability</h4>
                    <h2 style="color: {'#2E7D32' if conv_prob >= 0.5 else '#C62828'};">{conv_prob:.1%}</h2>
                    <p><b>Status:</b> {'Likely to Convert (High Intent)' if conv_flag == 1 else 'Unconverted / Low Intent'}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Expected Realized Sales</h4>
                    <h2 style="color: #1565C0;">₹ {pred_sales:,.2f}</h2>
                    <p><b>Model:</b> {sales_bundle['model_name']}</p>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# 6. STATISTICAL TESTING & METRICS
# ==============================================================================
elif app_mode == "Statistical Testing & Metrics":
    st.header("📐 Statistical Hypothesis Tests & Model Evaluation Metrics")
    
    tab1, tab2 = st.tabs(["Hypothesis Test Outcomes", "Model Benchmarks"])
    
    with tab1:
        st.subheader("Academic Statistical Hypothesis Verification")
        stats_file = os.path.join(PROJECT_ROOT, "outputs", "results", "statistical_tests.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                s_data = json.load(f)
            
            for k, test in s_data.items():
                with st.expander(f"📌 {k.replace('_', ' ')}"):
                    st.write(f"**Test Type:** {test.get('test_type')}")
                    st.write(f"**Null Hypothesis:** {test.get('null_hypothesis')}")
                    st.write(f"**Alternative Hypothesis:** {test.get('alternative_hypothesis')}")
                    p_val = test.get('p_value')
                    st.write(f"**p-value:** `{p_val:.5e}`")
                    st.markdown(f"**Decision:** :green[{test.get('decision')}]")
                    st.info(f"**Business Interpretation:** {test.get('business_interpretation')}")
                    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Regression Model Evaluation (Sales Prediction)")
            reg_f = os.path.join(PROJECT_ROOT, "outputs", "results", "regression_metrics.csv")
            if os.path.exists(reg_f):
                st.dataframe(pd.read_csv(reg_f), use_container_width=True)
                
        with c2:
            st.subheader("Classification Model Evaluation (Conversion Flag)")
            clf_f = os.path.join(PROJECT_ROOT, "outputs", "results", "classification_metrics.csv")
            if os.path.exists(clf_f):
                st.dataframe(pd.read_csv(clf_f), use_container_width=True)
                
        st.subheader("Top Feature Importances (Random Forest)")
        imp_f = os.path.join(PROJECT_ROOT, "outputs", "results", "top_feature_importance.csv")
        if os.path.exists(imp_f):
            st.dataframe(pd.read_csv(imp_f), use_container_width=True)

st.markdown("---")
st.caption("Tata Motors EV Marketing Analysis - MCA Capstone Project | Developed with Python, NumPy, SciPy & Streamlit")
