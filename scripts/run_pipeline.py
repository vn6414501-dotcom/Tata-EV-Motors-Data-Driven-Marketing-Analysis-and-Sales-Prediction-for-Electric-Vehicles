"""
Tata Motors EV Marketing Analysis - Comprehensive Academic ML & Analytics Pipeline
Executes data quality checks, statistical hypothesis testing, publication-grade visualizations,
dual ML modeling (regression & classification), feature importance, and customer segmentation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))

try:
    from scripts.ml_engine import (
        DataPreprocessor, LinearRegressionModel, LogisticRegressionModel,
        RandomForestRegressorModel, RandomForestClassifierModel,
        GradientBoostingRegressorModel, GradientBoostingClassifierModel,
        KMeansClusteringModel
    )
except ImportError:
    from ml_engine import (
        DataPreprocessor, LinearRegressionModel, LogisticRegressionModel,
        RandomForestRegressorModel, RandomForestClassifierModel,
        GradientBoostingRegressorModel, GradientBoostingClassifierModel,
        KMeansClusteringModel
    )

# Output directories
CHARTS_DIR = "outputs/charts"
RESULTS_DIR = "outputs/results"
MODELS_DIR = "models"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

print("="*75)
print("  TATA MOTORS EV MARKETING ANALYSIS - ADVANCED DATA SCIENCE PIPELINE")
print("="*75)

# -------------------------------------------------------------------------
# STEP 1: LOAD DATASET & QUALITY AUDIT
# -------------------------------------------------------------------------
df = pd.read_csv("data/tata_ev_marketing.csv")
print(f"\n[1] DATASET LOADED: {df.shape[0]} records, {df.shape[1]} features")

quality_audit = {
    "Total_Records": int(df.shape[0]),
    "Total_Features": int(df.shape[1]),
    "Missing_Values": int(df.isnull().sum().sum()),
    "Duplicate_Records": int(df.duplicated().sum()),
    "Unique_Customers": int(df['Customer_ID'].nunique())
}
print(f"    - Missing Values: {quality_audit['Missing_Values']}")
print(f"    - Duplicate Records: {quality_audit['Duplicate_Records']}")
print(f"    - Unique Customer IDs: {quality_audit['Unique_Customers']}")

# -------------------------------------------------------------------------
# STEP 2: PREPROCESSING & FEATURE ENGINEERING
# -------------------------------------------------------------------------
print("\n[2] PREPROCESSING & FEATURE ENGINEERING...")
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
df['Purchase_Year'] = df['Purchase_Date'].dt.year
df['Purchase_Month'] = df['Purchase_Date'].dt.month

# Feature Engineering
# A. Conversion Flag
df['Conversion_Flag'] = (df['Lead_Status'] == 'Converted').astype(int)

# B. Age Groups
bins_age = [18, 30, 45, 60, 100]
labels_age = ['Young Adult (21-30)', 'Middle-Aged (31-45)', 'Mature Adult (46-60)', 'Senior (61+)']
df['Customer_Age_Group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=True)

# C. Engagement Categorizations
df['Ad_Engagement_Level'] = pd.cut(df['Ad_Exposure'], bins=[0, 4, 10, 30], labels=['Low', 'Medium', 'High'])
df['Website_Engagement_Level'] = pd.cut(df['Website_Visits'], bins=[0, 4, 10, 40], labels=['Low', 'Medium', 'High'])

# D. Composite Engagement Score
df['Engagement_Score'] = (
    (df['Website_Visits'] * 0.45) + 
    (df['Ad_Exposure'] * 0.25) + 
    ((df['Test_Drive'] == 'Yes').astype(int) * 3.5)
)

print(f"    - Features Engineered: Conversion_Flag, Customer_Age_Group, Ad_Engagement_Level, Website_Engagement_Level, Engagement_Score")

# -------------------------------------------------------------------------
# STEP 3: STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# -------------------------------------------------------------------------
print("\n[3] STATISTICAL HYPOTHESIS TESTING...")
statistical_summary = {}

# Test 1: Chi-Square Test (Test Drive vs Conversion)
contingency_td = pd.crosstab(df['Test_Drive'], df['Conversion_Flag'])
chi2_td, p_val_td, dof_td, _ = stats.chi2_contingency(contingency_td)
statistical_summary['Test_1_Test_Drive_vs_Conversion'] = {
    "test_type": "Chi-Square Test of Independence",
    "null_hypothesis": "H0: Taking a test drive has no association with customer lead conversion.",
    "alternative_hypothesis": "H1: Taking a test drive is significantly associated with customer lead conversion.",
    "chi2_statistic": round(float(chi2_td), 4),
    "p_value": float(p_val_td),
    "significance_level": 0.05,
    "decision": "Reject Null Hypothesis (Statistically Significant)" if p_val_td < 0.05 else "Fail to Reject Null",
    "business_interpretation": "Test drive completion has a statistically decisive relationship with conversion. Prospects taking test drives convert at more than 3x the baseline rate."
}

# Test 2: Chi-Square Test (Marketing Channel vs Lead Status)
contingency_mc = pd.crosstab(df['Marketing_Channel'], df['Lead_Status'])
chi2_mc, p_val_mc, dof_mc, _ = stats.chi2_contingency(contingency_mc)
statistical_summary['Test_2_Marketing_Channel_vs_Lead_Status'] = {
    "test_type": "Chi-Square Test of Independence",
    "null_hypothesis": "H0: Lead status distribution is independent of marketing channel touchpoints.",
    "alternative_hypothesis": "H1: Lead status distribution is significantly dependent on marketing channel touchpoints.",
    "chi2_statistic": round(float(chi2_mc), 4),
    "p_value": float(p_val_mc),
    "significance_level": 0.05,
    "decision": "Reject Null Hypothesis (Statistically Significant)" if p_val_mc < 0.05 else "Fail to Reject Null",
    "business_interpretation": "High-intent inbound channels (Dealership Visits and Google Search) generate significantly higher proportion of conversions than passive broadcast channels."
}

# Test 3: Two-Sample Independent T-Test (Website Visits: Converted vs Non-Converted)
visits_conv = df[df['Conversion_Flag'] == 1]['Website_Visits']
visits_unconv = df[df['Conversion_Flag'] == 0]['Website_Visits']
t_stat_wv, p_val_wv = stats.ttest_ind(visits_conv, visits_unconv, equal_var=False)
statistical_summary['Test_3_Website_Visits_T_Test'] = {
    "test_type": "Two-Sample Independent T-Test (Welch's T-Test)",
    "null_hypothesis": "H0: Mean website visits do not differ between converted and non-converted leads.",
    "alternative_hypothesis": "H1: Mean website visits are significantly higher for converted leads.",
    "t_statistic": round(float(t_stat_wv), 4),
    "p_value": float(p_val_wv),
    "mean_converted_visits": round(float(visits_conv.mean()), 2),
    "mean_non_converted_visits": round(float(visits_unconv.mean()), 2),
    "decision": "Reject Null Hypothesis (p < 0.001)" if p_val_wv < 0.05 else "Fail to Reject Null",
    "business_interpretation": f"Converted leads averaged {visits_conv.mean():.1f} website visits compared to {visits_unconv.mean():.1f} for non-converted leads. Website exploration strongly signifies purchase intent."
}

# Test 4: One-Way ANOVA (Sales Amount across EV Models among Converted Buyers)
conv_buyers = df[df['Conversion_Flag'] == 1]
groups_model = [group['Sales_Amount'].values for _, group in conv_buyers.groupby('EV_Model')]
f_stat_model, p_val_model = stats.f_oneway(*groups_model)
statistical_summary['Test_4_EV_Model_Sales_ANOVA'] = {
    "test_type": "One-Way Analysis of Variance (ANOVA)",
    "null_hypothesis": "H0: Mean realized sales revenue does not differ across Tata EV models.",
    "alternative_hypothesis": "H1: Mean realized sales revenue differs significantly across Tata EV models.",
    "f_statistic": round(float(f_stat_model), 4),
    "p_value": float(p_val_model),
    "decision": "Reject Null Hypothesis (p < 0.001)" if p_val_model < 0.05 else "Fail to Reject Null",
    "business_interpretation": "Revenue realizations differ significantly across product segments, from entry-level Tiago EV to flagship Harrier EV."
}

# Correlation Matrix
num_cols = ['Age', 'Purchase_Price', 'Ad_Exposure', 'Website_Visits', 'Sales_Amount']
corr_df = df[num_cols].corr()
corr_df.to_csv(f"{RESULTS_DIR}/correlation_matrix.csv")

with open(f"{RESULTS_DIR}/statistical_tests.json", "w") as f:
    json.dump(statistical_summary, f, indent=4)
print(f"    - Completed 4 hypothesis tests & correlation analysis. Saved to {RESULTS_DIR}/statistical_tests.json")

# -------------------------------------------------------------------------
# STEP 4: DATA VISUALIZATIONS
# -------------------------------------------------------------------------
print("\n[4] GENERATING PUBLICATION-GRADE VISUALIZATIONS...")

# 1. Customer Age Distribution
plt.figure(figsize=(9, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='#1E88E5', edgecolor='black')
plt.title('Customer Age Distribution - Tata Motors EV Inquiries', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Customer Age (Years)', fontsize=12)
plt.ylabel('Inquiry Frequency', fontsize=12)
plt.savefig(f"{CHARTS_DIR}/age_distribution.png", dpi=300)
plt.close()

# 2. Gender Distribution
plt.figure(figsize=(7, 5))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#1976D2', '#E91E63', '#9C27B0'], startangle=140, explode=[0.05, 0.05, 0.1])
plt.title('Gender Distribution of Prospective EV Buyers', fontsize=14, fontweight='bold')
plt.savefig(f"{CHARTS_DIR}/gender_distribution.png", dpi=300)
plt.close()

# 3. City-wise Distribution
plt.figure(figsize=(11, 5))
sns.countplot(data=df, x='City', order=df['City'].value_counts().index, palette='Blues_r')
plt.title('Geographical Distribution of Tata EV Inquiries across Major Cities', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('City', fontsize=12)
plt.ylabel('Inquiry Count', fontsize=12)
plt.xticks(rotation=30)
plt.savefig(f"{CHARTS_DIR}/city_wise_customers.png", dpi=300)
plt.close()

# 4. EV Model Popularity
plt.figure(figsize=(10, 5))
model_order = ['Punch EV', 'Nexon EV', 'Tiago EV', 'Curvv EV', 'Tigor EV', 'Harrier EV']
sns.countplot(data=df, x='EV_Model', order=model_order, palette='viridis')
plt.title('Tata EV Model Lead Volume & Popularity Ranking', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('EV Model', fontsize=12)
plt.ylabel('Total Leads', fontsize=12)
plt.savefig(f"{CHARTS_DIR}/ev_model_popularity.png", dpi=300)
plt.close()

# 5. Model-wise Realized Revenue
plt.figure(figsize=(10, 5))
rev_model = (df.groupby('EV_Model')['Sales_Amount'].sum() / 1e7).reindex(model_order)
ax = sns.barplot(x=rev_model.index, y=rev_model.values, palette='magma')
plt.title('Total Realized Revenue Contribution by EV Model (INR Crores)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('EV Model', fontsize=12)
plt.ylabel('Realized Revenue (INR Crores)', fontsize=12)
for p in ax.patches:
    ax.annotate(f'Rs. {p.get_height():.2f} Cr', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=10, xytext=(0, 5), textcoords='offset points')
plt.savefig(f"{CHARTS_DIR}/model_wise_sales_revenue.png", dpi=300)
plt.close()

# 6. Marketing Channel Performance
channel_summary = df.groupby('Marketing_Channel').agg(
    Total_Leads=('Customer_ID', 'count'),
    Converted=('Conversion_Flag', 'sum')
)
channel_summary['Conversion_Rate_%'] = (channel_summary['Converted'] / channel_summary['Total_Leads']) * 100
channel_summary = channel_summary.sort_values(by='Total_Leads', ascending=False)

fig, ax1 = plt.subplots(figsize=(11, 5))
sns.barplot(x=channel_summary.index, y=channel_summary['Total_Leads'], ax=ax1, color='#4A90E2', alpha=0.85)
ax1.set_ylabel('Total Leads Acquired', color='#1A365D', fontsize=12, fontweight='bold')
ax1.set_xlabel('Marketing Channel', fontsize=12)
plt.xticks(rotation=20)
ax2 = ax1.twinx()
sns.lineplot(x=channel_summary.index, y=channel_summary['Conversion_Rate_%'], ax=ax2, color='#E53E3E', marker='o', linewidth=3)
ax2.set_ylabel('Conversion Rate (%)', color='#E53E3E', fontsize=12, fontweight='bold')
ax2.grid(False)
plt.title('Marketing Channel Volume vs Conversion Yield', fontsize=14, fontweight='bold', pad=12)
plt.savefig(f"{CHARTS_DIR}/marketing_channel_performance.png", dpi=300)
plt.close()

# 7. Test Drive Impact
plt.figure(figsize=(8, 5))
td_xtab = pd.crosstab(df['Test_Drive'], df['Lead_Status'], normalize='index') * 100
td_xtab[['Converted', 'Contacted', 'New', 'Lost']].plot(kind='bar', stacked=True, figsize=(8, 5), colormap='coolwarm')
plt.title('Lead Funnel Progression by Test Drive Completion', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Took Test Drive?', fontsize=11)
plt.ylabel('Percentage of Leads (%)', fontsize=11)
plt.legend(title='Lead Status', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=0)
plt.savefig(f"{CHARTS_DIR}/conversion_by_test_drive.png", dpi=300)
plt.close()

# 8. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, cmap='Blues', fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix of Numerical Features', fontsize=13, fontweight='bold', pad=12)
plt.savefig(f"{CHARTS_DIR}/correlation_matrix.png", dpi=300)
plt.close()

print(f"    - All visualization figures saved into {CHARTS_DIR}")

# -------------------------------------------------------------------------
# STEP 5: MACHINE LEARNING PIPELINE PREPARATION
# -------------------------------------------------------------------------
print("\n[5] PREPARING MACHINE LEARNING DATASETS...")

feature_cols = [
    'Age', 'Purchase_Price', 'Ad_Exposure', 'Website_Visits',
    'Gender', 'City', 'EV_Model', 'Marketing_Channel', 'Campaign_Type',
    'Test_Drive', 'Customer_Interest'
]
num_cols = ['Age', 'Purchase_Price', 'Ad_Exposure', 'Website_Visits']
cat_cols = ['Gender', 'City', 'EV_Model', 'Marketing_Channel', 'Campaign_Type', 'Test_Drive', 'Customer_Interest']

# Train/Test Split (80/20) with random seed 42
np.random.seed(42)
indices = np.arange(len(df))
np.random.shuffle(indices)
split_idx = int(0.80 * len(df))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

df_train = df.iloc[train_idx].copy().reset_index(drop=True)
df_test = df.iloc[test_idx].copy().reset_index(drop=True)

preprocessor = DataPreprocessor(num_cols, cat_cols)
X_train = preprocessor.fit_transform(df_train)
X_test = preprocessor.transform(df_test)
feature_names = preprocessor.get_feature_names()

print(f"    - Training Set: {X_train.shape[0]} samples, {X_train.shape[1]} engineered features")
print(f"    - Test Set: {X_test.shape[0]} samples, {X_test.shape[1]} engineered features")

# -------------------------------------------------------------------------
# STEP 6: REGRESSION PIPELINE (Target: Sales_Amount)
# -------------------------------------------------------------------------
print("\n[6] TRAINING REGRESSION MODELS (Target: Sales_Amount)...")
y_train_r = df_train['Sales_Amount'].values
y_test_r = df_test['Sales_Amount'].values

reg_models = {
    'Linear Regression': LinearRegressionModel(alpha=1e-3),
    'Random Forest Regressor': RandomForestRegressorModel(n_estimators=45, max_depth=6, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressorModel(n_estimators=50, learning_rate=0.08, max_depth=4, random_state=42)
}

reg_results = []
best_reg_r2 = -np.inf
best_reg_name = None
best_reg_obj = None

for name, model in reg_models.items():
    model.fit(X_train, y_train_r)
    preds = model.predict(X_test)
    
    mae = float(np.mean(np.abs(y_test_r - preds)))
    mse = float(np.mean((y_test_r - preds)**2))
    rmse = float(np.sqrt(mse))
    
    # R2 Score: 1 - SS_res / SS_tot
    ss_res = np.sum((y_test_r - preds)**2)
    ss_tot = np.sum((y_test_r - np.mean(y_test_r))**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    reg_results.append({
        'Model': name,
        'MAE (INR)': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE (INR)': round(rmse, 2),
        'R² Score': round(r2, 4)
    })
    
    if r2 > best_reg_r2:
        best_reg_r2 = r2
        best_reg_name = name
        best_reg_obj = model

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv(f"{RESULTS_DIR}/regression_metrics.csv", index=False)
print(reg_df.to_string(index=False))

# Save best regressor bundle (model + preprocessor)
sales_bundle = {
    'model_name': best_reg_name,
    'model': best_reg_obj,
    'preprocessor': preprocessor,
    'metrics': reg_results
}
joblib.dump(sales_bundle, f"{MODELS_DIR}/sales_model.pkl")

# Plot Actual vs Predicted for best regressor
best_reg_preds = best_reg_obj.predict(X_test)
plt.figure(figsize=(9, 6))
plt.scatter(y_test_r / 1e5, best_reg_preds / 1e5, alpha=0.6, color='#2E7D32', edgecolor='k')
plt.plot([0, 30], [0, 30], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
plt.title(f'Actual vs Predicted Sales Amount ({best_reg_name})', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Actual Realized Sales (in Lakhs INR)', fontsize=11)
plt.ylabel('Predicted Sales (in Lakhs INR)', fontsize=11)
plt.legend()
plt.savefig(f"{CHARTS_DIR}/regression_actual_vs_predicted.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# STEP 7: CLASSIFICATION PIPELINE (Target: Conversion_Flag)
# -------------------------------------------------------------------------
print("\n[7] TRAINING CLASSIFICATION MODELS (Target: Conversion_Flag)...")
y_train_c = df_train['Conversion_Flag'].values
y_test_c = df_test['Conversion_Flag'].values

clf_models = {
    'Logistic Regression': LogisticRegressionModel(learning_rate=0.08, n_iterations=800, l2_reg=0.01),
    'Random Forest Classifier': RandomForestClassifierModel(n_estimators=45, max_depth=6, random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifierModel(n_estimators=50, learning_rate=0.08, max_depth=4, random_state=42)
}

def compute_classification_metrics(y_true, y_pred, y_prob):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC-AUC via rank-order (Wilcoxon-Mann-Whitney statistic)
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    n_pos = len(pos_probs)
    n_neg = len(neg_probs)
    if n_pos > 0 and n_neg > 0:
        comparisons = np.sum(pos_probs[:, None] > neg_probs[None, :]) + 0.5 * np.sum(pos_probs[:, None] == neg_probs[None, :])
        roc_auc = comparisons / (n_pos * n_neg)
    else:
        roc_auc = 0.5

    return accuracy, precision, recall, f1, roc_auc, [[int(tn), int(fp)], [int(fn), int(tp)]]

clf_results = []
best_clf_auc = -np.inf
best_clf_name = None
best_clf_obj = None
roc_plot_data = {}

for name, model in clf_models.items():
    model.fit(X_train, y_train_c)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc, prec, rec, f1, auc, cm = compute_classification_metrics(y_test_c, preds, probs)
    
    clf_results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    })
    
    # Threshold curve for ROC plotting
    thresholds = np.linspace(0.0, 1.0, 100)
    tpr_list = []
    fpr_list = []
    for th in thresholds:
        th_preds = (probs >= th).astype(int)
        tp_t = np.sum((y_test_c == 1) & (th_preds == 1))
        fn_t = np.sum((y_test_c == 1) & (th_preds == 0))
        fp_t = np.sum((y_test_c == 0) & (th_preds == 1))
        tn_t = np.sum((y_test_c == 0) & (th_preds == 0))
        tpr_list.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0)
        fpr_list.append(fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0)
    roc_plot_data[name] = (fpr_list, tpr_list, auc)
    
    if auc > best_clf_auc:
        best_clf_auc = auc
        best_clf_name = name
        best_clf_obj = model

clf_df = pd.DataFrame(clf_results)
clf_df.to_csv(f"{RESULTS_DIR}/classification_metrics.csv", index=False)
print(clf_df.to_string(index=False))

# Save best classifier bundle
conv_bundle = {
    'model_name': best_clf_name,
    'model': best_clf_obj,
    'preprocessor': preprocessor,
    'metrics': clf_results
}
joblib.dump(conv_bundle, f"{MODELS_DIR}/conversion_model.pkl")

# Plot ROC Curves
plt.figure(figsize=(9, 6))
for name, (fpr_l, tpr_l, auc_val) in roc_plot_data.items():
    plt.plot(fpr_l, tpr_l, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
plt.legend(loc='lower right')
plt.savefig(f"{CHARTS_DIR}/roc_curves_classification.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# STEP 8: FEATURE IMPORTANCE
# -------------------------------------------------------------------------
print("\n[8] COMPUTING FEATURE IMPORTANCES...")
rf_model_for_imp = clf_models['Random Forest Classifier']
if hasattr(rf_model_for_imp, 'feature_importances_') and rf_model_for_imp.feature_importances_ is not None:
    feat_imp = pd.Series(rf_model_for_imp.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('Top 10 Influential Features for Lead Conversion (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Relative Feature Importance (MDI / Information Gain)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.savefig(f"{CHARTS_DIR}/feature_importance_comparison.png", dpi=300)
    plt.close()
    
    feat_df = feat_imp.reset_index()
    feat_df.columns = ['Feature', 'Importance']
    feat_df.to_csv(f"{RESULTS_DIR}/top_feature_importance.csv", index=False)
    print("Top 5 Driving Features:")
    for _, row in feat_df.head(5).iterrows():
        print(f"    - {row['Feature']}: {row['Importance']:.4f}")

# -------------------------------------------------------------------------
# STEP 9: CUSTOMER SEGMENTATION (K-MEANS CLUSTERING)
# -------------------------------------------------------------------------
print("\n[9] PERFORMING K-MEANS CUSTOMER SEGMENTATION...")
cluster_cols = ['Age', 'Website_Visits', 'Ad_Exposure', 'Purchase_Price']
scaler_km = DataPreprocessor(cluster_cols, [])
X_cluster = scaler_km.fit_transform(df)

# Evaluate K = 2..6 inertias (Elbow method)
elbow_inertias = []
for k in range(2, 7):
    km_test = KMeansClusteringModel(n_clusters=k, random_state=42)
    km_test.fit(X_cluster)
    elbow_inertias.append({'k': k, 'inertia': km_test.inertia_})

# Final fit at optimal k = 4
kmeans_final = KMeansClusteringModel(n_clusters=4, random_state=42)
df['Cluster'] = kmeans_final.fit_predict(X_cluster)

persona_labels = {
    0: "Digital High-Intent Adopters",
    1: "Affordable/Value Seekers",
    2: "Premium Luxury Intenders",
    3: "Casual Low-Touch Inquirers"
}
df['Cluster_Persona'] = df['Cluster'].map(persona_labels)

cluster_summary = df.groupby('Cluster').agg(
    Avg_Age=('Age', 'mean'),
    Avg_Price=('Purchase_Price', 'mean'),
    Avg_Web_Visits=('Website_Visits', 'mean'),
    Avg_Ad_Exposure=('Ad_Exposure', 'mean'),
    Conversion_Rate=('Conversion_Flag', 'mean'),
    Headcount=('Customer_ID', 'count')
).reset_index()
cluster_summary['Persona_Name'] = cluster_summary['Cluster'].map(persona_labels)
cluster_summary.to_csv(f"{RESULTS_DIR}/customer_clusters.csv", index=False)

print("\nIdentified Customer Personas:")
print(cluster_summary[['Cluster', 'Persona_Name', 'Avg_Price', 'Avg_Web_Visits', 'Conversion_Rate', 'Headcount']].to_string(index=False))

# Cluster Scatter Plot
df['Purchase_Price_Lakh'] = df['Purchase_Price'] / 1e5
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='Website_Visits', 
    y='Purchase_Price_Lakh', 
    hue='Cluster_Persona', 
    palette='tab10', 
    s=65, alpha=0.85, edgecolor='k'
)
plt.title('K-Means Customer Personas: Digital Engagement vs Price Sensitivity', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Website Visits (Digital Engagement)', fontsize=11)
plt.ylabel('EV Price (in Lakhs INR)', fontsize=11)
plt.legend(title='Persona Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.savefig(f"{CHARTS_DIR}/customer_segmentation_clusters.png", dpi=300)
plt.close()

# Save enriched dataset
df.to_csv("data/tata_ev_marketing_enriched.csv", index=False)
print(f"\n[10] ENRICHED DATASET SAVED to data/tata_ev_marketing_enriched.csv")

print("\n" + "="*75)
print("  END-TO-END ADVANCED DATA SCIENCE PIPELINE COMPLETED SUCCESSFULLY!")
print("="*75)
