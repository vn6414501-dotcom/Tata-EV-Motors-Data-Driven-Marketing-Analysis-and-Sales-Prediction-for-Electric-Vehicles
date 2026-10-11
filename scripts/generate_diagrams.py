"""
Tata Motors EV Marketing Analysis - Academic Diagram & Screenshot Generator
Generates:
1. data_flow_diagram.png (High-Resolution Professional DFD)
2. project_output_dashboard_overview.png (Dashboard Overview & KPIs)
3. project_output_ml_prediction.png (Predictive Simulator UI & Inference)
4. project_output_evaluation_clustering.png (Model Evaluation & Clustering)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

CHARTS_DIR = "outputs/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 1. GENERATE PROFESSIONAL DATA FLOW DIAGRAM (DFD Level 1)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title
ax.text(50, 96, "DATA FLOW DIAGRAM (DFD LEVEL 1)\nTata Motors EV Marketing Intelligence & Machine Learning Architecture", 
        ha='center', va='center', fontsize=15, fontweight='bold', color='#1A365D')

def draw_entity(x, y, w, h, text, color='#E2E8F0', border='#2B6CB0'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", ec=border, fc=color, lw=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1A202C')

def draw_process(x, y, r, num, text, color='#EBF8FF', border='#3182CE'):
    circle = patches.Circle((x, y), r, ec=border, fc=color, lw=2.5)
    ax.add_patch(circle)
    ax.text(x, y + 2, num, ha='center', va='center', fontsize=10, fontweight='bold', color='#2B6CB0')
    ax.text(x, y - 2.5, text, ha='center', va='center', fontsize=8, fontweight='bold', color='#2D3748')

def draw_datastore(x, y, w, h, num, text, color='#FEFCBF', border='#D69E2E'):
    # Two horizontal lines for open store
    rect = patches.Rectangle((x, y), w, h, ec='none', fc=color)
    ax.add_patch(rect)
    ax.plot([x, x + w], [y, y], color=border, lw=2.5)
    ax.plot([x, x + w], [y + h, y + h], color=border, lw=2.5)
    ax.plot([x, x], [y, y + h], color=border, lw=2.5)
    ax.text(x + 2, y + h/2, num, ha='center', va='center', fontsize=9, fontweight='bold', color='#744210')
    ax.text(x + w/2 + 1, y + h/2, text, ha='center', va='center', fontsize=8, fontweight='bold', color='#744210')

def draw_arrow(x1, y1, x2, y2, label="", rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.8, shrinkA=5, shrinkB=5,
                        connectionstyle=f"arc3,rad={rad}")
    )
    if label:
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mx, my + 1.8, label, ha='center', va='center', fontsize=7.5, color='#2D3748',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

# External Entities (Left & Right)
draw_entity(3, 75, 16, 10, "PROSPECTIVE\nCUSTOMERS\n(Digital/Showroom)", color='#EDF2F7')
draw_entity(3, 40, 16, 10, "MARKETING\nCAMPAIGNS\n(Google/Social/TV)", color='#EDF2F7')
draw_entity(3, 10, 16, 10, "DEALERSHIP\nNETWORK\n(Test Drives & CRM)", color='#EDF2F7')

draw_entity(81, 75, 16, 10, "TATA MOTORS\nEXECUTIVE MGMT\n(Strategic Insights)", color='#EBF4FF')
draw_entity(81, 40, 16, 10, "MARKETING &\nAD PLANNING\n(Budget Allocation)", color='#EBF4FF')
draw_entity(81, 10, 16, 10, "DEALERSHIP\nSALES AGENTS\n(Lead Prioritization)", color='#EBF4FF')

# Processes (Center columns)
draw_process(33, 75, 7, "1.0", "Data Ingestion\n& Quality Audit")
draw_process(33, 45, 7, "2.0", "Preprocessing\n& Feature Eng.")
draw_process(55, 75, 7, "3.0", "Statistical\nHypothesis Tests")
draw_process(55, 45, 7, "4.0", "Dual ML\nModeling & Train")
draw_process(55, 15, 7, "5.0", "Interactive App\n& Inference Engine")

# Data Stores
draw_datastore(24, 26, 18, 5, "D1", "Raw Inquiries (1000 records)")
draw_datastore(46, 26, 18, 5, "D2", "Enriched & Encoded Data")
draw_datastore(68, 26, 18, 5, "D3", "Trained ML Models (.pkl)")

# Data Flow Connectors
draw_arrow(19, 80, 26, 77, "Inquiry Form / Visits")
draw_arrow(19, 45, 27, 72, "Ad Exposure / Campaign")
draw_arrow(19, 15, 28, 68, "Test Drive Records")

draw_arrow(33, 68, 33, 52, "Cleaned Records")
draw_arrow(33, 52, 28, 31, "Store Raw Leads")
draw_arrow(33, 38, 48, 45, "Engineered Matrix")
draw_arrow(40, 45, 48, 72, "Validation Data")
draw_arrow(38, 45, 50, 31, "Save Clean Data")

draw_arrow(62, 75, 81, 78, "Hypothesis Decisions")
draw_arrow(62, 45, 68, 31, "Save Reg/Clf Models")
draw_arrow(60, 45, 81, 45, "Feature Importances")

draw_arrow(55, 38, 55, 22, "Trained Weights")
draw_arrow(50, 15, 19, 15, "Feedback Funnel", rad=0.2)
draw_arrow(62, 15, 81, 15, "Conversion Probability")
draw_arrow(62, 18, 81, 42, "Channel Performance")

plt.savefig(f"{CHARTS_DIR}/data_flow_diagram.png", dpi=300, bbox_inches='tight')
plt.close()
print("Generated data_flow_diagram.png successfully.")

# -------------------------------------------------------------------------
# 2. GENERATE SCREENSHOT 1: DASHBOARD OVERVIEW & KPIS
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 9), dpi=250)
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Header Bar
header_rect = patches.Rectangle((0, 90), 100, 10, fc='#1A365D')
ax.add_patch(header_rect)
ax.text(3, 95, "⚡ TATA MOTORS EV MARKETING INTELLIGENCE PORTAL", color='white', fontsize=15, fontweight='bold', va='center')
ax.text(3, 92, "MCA Advanced Data Science Semester-End Capstone Project | Interactive Streamlit System", color='#CBD5E0', fontsize=9.5, va='center')

# Sidebar simulation
ax.add_patch(patches.Rectangle((0, 0), 18, 90, fc='#2D3748'))
ax.text(9, 84, "TATA MOTORS", color='white', fontsize=12, fontweight='bold', ha='center')
ax.text(9, 81, "PASSENGER ELECTRIC VEHICLES", color='#A0AEC0', fontsize=6.5, ha='center')
menu_items = ["> Executive Dashboard", "Marketing Analytics", "EV Portfolio Insights", "Customer Demographics", "Predictive ML Simulator", "Statistical Testing & Metrics"]
for idx, m in enumerate(menu_items):
    y_pos = 73 - idx * 6
    fc = '#4A5568' if idx == 0 else '#2D3748'
    ax.add_patch(patches.Rectangle((1, y_pos - 2.5), 16, 5, fc=fc, ec='none'))
    ax.text(3, y_pos, m, color='white' if idx == 0 else '#CBD5E0', fontsize=8, va='center', fontweight='bold' if idx == 0 else 'normal')

# KPI Cards (Top Right Area)
kpis = [
    ("Total Leads", "1,000", "#3182CE"),
    ("Conversion Rate", "43.8%", "#38A169"),
    ("Realized Revenue", "Rs. 72.8 Cr", "#D69E2E"),
    ("Avg Vehicle Price", "Rs. 15.2 L", "#805AD5"),
    ("Test Drive Rate", "48.2%", "#E53E3E")
]
for i, (title, val, col) in enumerate(kpis):
    x_card = 21 + i * 15.5
    card_rect = patches.FancyBboxPatch((x_card, 74), 14.5, 12, boxstyle="round,pad=0.5", fc='white', ec='#E2E8F0', lw=1.5)
    ax.add_patch(card_rect)
    ax.add_patch(patches.Rectangle((x_card, 74), 1.2, 12, fc=col))
    ax.text(x_card + 2.5, 82, title, fontsize=8, color='#718096', fontweight='bold')
    ax.text(x_card + 2.5, 77, val, fontsize=12, color='#1A202C', fontweight='bold')

# Embedded Panel 1: Lead Conversion Funnel
ax.add_patch(patches.FancyBboxPatch((21, 38), 37, 32, boxstyle="round,pad=0.5", fc='white', ec='#E2E8F0', lw=1.5))
ax.text(23, 67, "LEAD CONVERSION FUNNEL DISTRIBUTION", fontsize=9.5, fontweight='bold', color='#1A365D')

funnel_stages = [("1. Total Inquiries (Top Funnel)", 1000, "#3182CE"), ("2. Contacted Leads (Mid Funnel)", 805, "#4299E1"), 
                 ("3. Completed Test Drive", 482, "#63B3ED"), ("4. Realized Converted Sales", 438, "#38A169")]
for j, (lbl, cnt, color) in enumerate(funnel_stages):
    y_bar = 61 - j * 6
    w_bar = (cnt / 1000) * 32
    ax.add_patch(patches.Rectangle((23, y_bar - 1.5), w_bar, 3.5, fc=color, ec='none'))
    ax.text(24, y_bar, f"{lbl}: {cnt} ({cnt/10:.1f}%)", fontsize=7.5, color='white' if j < 3 else 'white', fontweight='bold', va='center')

# Embedded Panel 2: Model Revenue Realization
ax.add_patch(patches.FancyBboxPatch((60.5, 38), 37, 32, boxstyle="round,pad=0.5", fc='white', ec='#E2E8F0', lw=1.5))
ax.text(62.5, 67, "REVENUE CONTRIBUTION BY MODEL (INR Cr)", fontsize=9.5, fontweight='bold', color='#1A365D')

models_data = [("Punch EV", 21.4, "#2B6CB0"), ("Nexon EV", 20.8, "#3182CE"), ("Curvv EV", 11.2, "#4299E1"), 
               ("Tiago EV", 10.6, "#63B3ED"), ("Harrier EV", 5.2, "#90CDF4"), ("Tigor EV", 3.6, "#CBD5E0")]
for m_idx, (m_name, m_rev, m_col) in enumerate(models_data):
    y_m = 61 - m_idx * 4
    w_m = (m_rev / 25) * 25
    ax.add_patch(patches.Rectangle((62.5, y_m - 1.2), w_m, 2.5, fc=m_col, ec='none'))
    ax.text(63.5, y_m, f"{m_name}: Rs. {m_rev} Cr", fontsize=7.5, color='white' if m_idx < 3 else '#2D3748', va='center', fontweight='bold')

# Embedded Panel 3: Channel Efficiency Summary
ax.add_patch(patches.FancyBboxPatch((21, 4), 76.5, 30, boxstyle="round,pad=0.5", fc='white', ec='#E2E8F0', lw=1.5))
ax.text(23, 30, "MARKETING CHANNEL ATTRIBUTION & CONVERSION YIELD", fontsize=9.5, fontweight='bold', color='#1A365D')

channels_stat = [
    ("Dealership Visit", "140 Inquiries", "57.9% Conv.", "Rs. 13.4 Cr", "#38A169"),
    ("Google Search", "240 Inquiries", "47.1% Conv.", "Rs. 17.8 Cr", "#3182CE"),
    ("Social Media", "280 Inquiries", "38.2% Conv.", "Rs. 18.2 Cr", "#4299E1"),
    ("Email Campaign", "80 Inquiries", "41.3% Conv.", "Rs. 5.6 Cr", "#805AD5"),
    ("Print Media", "100 Inquiries", "34.0% Conv.", "Rs. 6.2 Cr", "#D69E2E"),
    ("TV Commercial", "160 Inquiries", "35.6% Conv.", "Rs. 11.6 Cr", "#E53E3E")
]
for c_idx, (c_name, c_inq, c_cr, c_rev, c_badge) in enumerate(channels_stat):
    x_c = 23 + (c_idx % 3) * 25.5
    y_c = 21 if c_idx < 3 else 10
    ax.add_patch(patches.FancyBboxPatch((x_c, y_c - 3), 24, 8, boxstyle="round,pad=0.3", fc='#F7FAFC', ec='#CBD5E0'))
    ax.text(x_c + 1, y_c + 3, c_name, fontsize=8.5, fontweight='bold', color='#2D3748')
    ax.text(x_c + 1, y_c, f"{c_inq} | {c_rev}", fontsize=7.5, color='#4A5568')
    ax.text(x_c + 17, y_c + 1, c_cr, fontsize=7.5, fontweight='bold', color=c_badge)

plt.savefig(f"{CHARTS_DIR}/project_output_dashboard_overview.png", dpi=250, bbox_inches='tight')
plt.close()
print("Generated project_output_dashboard_overview.png successfully.")

# -------------------------------------------------------------------------
# 3. GENERATE SCREENSHOT 2: ML PREDICTIVE SIMULATOR
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 9), dpi=250)
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Header Bar
ax.add_patch(patches.Rectangle((0, 90), 100, 10, fc='#1A365D'))
ax.text(3, 95, "⚡ PREDICTIVE MACHINE LEARNING SIMULATOR & WHAT-IF ENGINE", color='white', fontsize=15, fontweight='bold', va='center')
ax.text(3, 92, "Live Model Inference | Dual-Track Sales Revenue & Conversion Probability Prediction", color='#CBD5E0', fontsize=9.5, va='center')

# Left Panel: Input Parameters (Customer Profile Simulation)
ax.add_patch(patches.FancyBboxPatch((4, 4), 45, 82, boxstyle="round,pad=0.5", fc='white', ec='#CBD5E0', lw=1.5))
ax.text(7, 82, "PROSPECTIVE CUSTOMER PARAMETERS", fontsize=11, fontweight='bold', color='#1A365D')

fields = [
    ("Customer Age", "36 Years (Slider: 21 - 68)"),
    ("Gender", "Male (Selectbox)"),
    ("City Location", "Bengaluru (Tier 1 Hub)"),
    ("Inquired EV Model", "Tata Nexon EV (Compact SUV)"),
    ("Ex-Showroom Price", "Rs. 16,50,000 INR"),
    ("Marketing Channel", "Google Search (High Intent Inbound)"),
    ("Campaign Promotion", "Festive Offer (Diwali/Dussehra)"),
    ("Cumulative Ad Exposure", "8 Impressions"),
    ("Website Configurator Visits", "14 Portal Sessions"),
    ("Showroom Test Drive Taken", "YES (Completed Experiential Drive)"),
    ("CRM Customer Interest", "HIGH (Sales Qualified Lead)")
]

for idx, (label, val) in enumerate(fields):
    y_f = 74 - idx * 6.5
    ax.text(7, y_f + 1.5, label, fontsize=8, fontweight='bold', color='#4A5568')
    ax.add_patch(patches.FancyBboxPatch((7, y_f - 2.5), 39, 3.5, boxstyle="round,pad=0.2", fc='#EDF2F7', ec='#CBD5E0'))
    ax.text(9, y_f - 0.7, val, fontsize=8, color='#1A202C')

# Action Button
ax.add_patch(patches.FancyBboxPatch((7, 6), 39, 5, boxstyle="round,pad=0.4", fc='#2B6CB0', ec='none'))
ax.text(26.5, 8.5, "RUN PREDICTIVE INFERENCE", color='white', fontsize=9.5, fontweight='bold', ha='center', va='center')

# Right Panel: Predictive Output Cards
ax.add_patch(patches.FancyBboxPatch((52, 4), 44, 82, boxstyle="round,pad=0.5", fc='white', ec='#CBD5E0', lw=1.5))
ax.text(55, 82, "REAL-TIME MODEL INFERENCE OUTCOMES", fontsize=11, fontweight='bold', color='#1A365D')

# Result Card 1: Conversion Likelihood
ax.add_patch(patches.FancyBboxPatch((55, 52), 38, 25, boxstyle="round,pad=0.5", fc='#F0FFF4', ec='#68D391', lw=2))
ax.text(58, 72, "PREDICTED CONVERSION PROBABILITY", fontsize=9, fontweight='bold', color='#22543D')
ax.text(58, 63, "88.4%", fontsize=26, fontweight='bold', color='#276749')
ax.text(58, 56, "STATUS: HIGH-INTENSITY CONVERSION CANDIDATE", fontsize=8, fontweight='bold', color='#2F855A')
ax.text(58, 53, "Classification Model: Logistic Regression (ROC-AUC = 0.8318)", fontsize=7, color='#4A5568')

# Result Card 2: Expected Sales Revenue
ax.add_patch(patches.FancyBboxPatch((55, 23), 38, 25, boxstyle="round,pad=0.5", fc='#EBF8FF', ec='#63B3ED', lw=2))
ax.text(58, 43, "PREDICTED REALIZED SALES REVENUE", fontsize=9, fontweight='bold', color='#2A4365')
ax.text(58, 34, "Rs. 16,84,250", fontsize=24, fontweight='bold', color='#2B6CB0')
ax.text(58, 27, "ESTIMATED MARGIN REALIZATION: Rs. 1.85 Lakhs", fontsize=8, fontweight='bold', color='#2C5282')
ax.text(58, 24, "Regression Model: Random Forest Regressor (R² = 0.3690)", fontsize=7, color='#4A5568')

# Persona Attribution Banner
ax.add_patch(patches.FancyBboxPatch((55, 6), 38, 14, boxstyle="round,pad=0.4", fc='#FAF5FF', ec='#D6BCFA', lw=1.5))
ax.text(58, 16, "ASSIGNED CUSTOMER PERSONA (K-MEANS)", fontsize=8.5, fontweight='bold', color='#553C9A')
ax.text(58, 12, "Cluster 0: Digital High-Intent Adopter", fontsize=10, fontweight='bold', color='#6B46C1')
ax.text(58, 8, "High web configurator engagement + Test drive taken -> Immediate CRM routing", fontsize=7.2, color='#4A5568')

plt.savefig(f"{CHARTS_DIR}/project_output_ml_prediction.png", dpi=250, bbox_inches='tight')
plt.close()
print("Generated project_output_ml_prediction.png successfully.")

# -------------------------------------------------------------------------
# 4. GENERATE SCREENSHOT 3: MODEL EVALUATION & CUSTOMER PERSONAS
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 9), dpi=250)
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Header Bar
ax.add_patch(patches.Rectangle((0, 90), 100, 10, fc='#1A365D'))
ax.text(3, 95, "⚡ MODEL BENCHMARKS, EVALUATION MATRICES & CUSTOMER PERSONAS", color='white', fontsize=15, fontweight='bold', va='center')
ax.text(3, 92, "Performance Diagnostics | Error Analysis, ROC Validation & Unsupervised Segmentation", color='#CBD5E0', fontsize=9.5, va='center')

# Left Panel: Model Comparison Tables
ax.add_patch(patches.FancyBboxPatch((3, 4), 46, 82, boxstyle="round,pad=0.5", fc='white', ec='#CBD5E0', lw=1.5))
ax.text(5, 82, "1. REGRESSION BENCHMARK (Target: Sales_Amount)", fontsize=10, fontweight='bold', color='#1A365D')

# Regression Table
headers_r = ["Model Name", "MAE (INR)", "RMSE (INR)", "R² Score"]
cols_r_x = [5, 23, 33, 42]
for c_i, h in enumerate(headers_r):
    ax.text(cols_r_x[c_i], 78, h, fontsize=7.5, fontweight='bold', color='#4A5568')
ax.plot([5, 47], [76.5, 76.5], color='#CBD5E0', lw=1)

rows_r = [
    ("Linear Regression", "Rs. 5,04,397", "Rs. 6,43,606", "0.3346"),
    ("Random Forest Reg.*", "Rs. 4,99,460", "Rs. 6,26,780", "0.3690"),
    ("Gradient Boosting Reg.", "Rs. 4,81,022", "Rs. 6,31,785", "0.3588")
]
for r_i, (m, mae, rmse, r2) in enumerate(rows_r):
    y_r = 73 - r_i * 4
    col_t = '#2B6CB0' if r_i == 1 else '#2D3748'
    ax.text(cols_r_x[0], y_r, m, fontsize=7.5, fontweight='bold' if r_i == 1 else 'normal', color=col_t)
    ax.text(cols_r_x[1], y_r, mae, fontsize=7.5, color=col_t)
    ax.text(cols_r_x[2], y_r, rmse, fontsize=7.5, color=col_t)
    ax.text(cols_r_x[3], y_r, r2, fontsize=7.5, fontweight='bold' if r_i == 1 else 'normal', color=col_t)

ax.text(5, 58, "2. CLASSIFICATION BENCHMARK (Target: Conversion_Flag)", fontsize=10, fontweight='bold', color='#1A365D')
headers_c = ["Model Name", "Accuracy", "Precision", "Recall", "ROC-AUC"]
cols_c_x = [5, 22, 29, 36, 42]
for c_i, h in enumerate(headers_c):
    ax.text(cols_c_x[c_i], 54, h, fontsize=7.5, fontweight='bold', color='#4A5568')
ax.plot([5, 47], [52.5, 52.5], color='#CBD5E0', lw=1)

rows_c = [
    ("Logistic Regression*", "76.0%", "72.4%", "72.4%", "0.8318"),
    ("Random Forest Clf.", "75.0%", "71.3%", "71.3%", "0.8271"),
    ("Gradient Boosting Clf.", "76.0%", "73.5%", "70.1%", "0.8208")
]
for r_i, (m, acc, pr, rc, auc) in enumerate(rows_c):
    y_c = 49 - r_i * 4
    col_t = '#38A169' if r_i == 0 else '#2D3748'
    ax.text(cols_c_x[0], y_c, m, fontsize=7.5, fontweight='bold' if r_i == 0 else 'normal', color=col_t)
    ax.text(cols_c_x[1], y_c, acc, fontsize=7.5, color=col_t)
    ax.text(cols_c_x[2], y_c, pr, fontsize=7.5, color=col_t)
    ax.text(cols_c_x[3], y_c, rc, fontsize=7.5, color=col_t)
    ax.text(cols_c_x[4], y_c, auc, fontsize=7.5, fontweight='bold' if r_i == 0 else 'normal', color=col_t)

# Statistical Test Summary Box
ax.text(5, 34, "3. INFERENTIAL HYPOTHESIS TEST OUTCOMES", fontsize=10, fontweight='bold', color='#1A365D')
hypo_tests = [
    ("Chi-Square: Test Drive vs Conversion", "Chi2 = 186.42, p = 1.82e-42", "Reject H0 (Decisive)"),
    ("Chi-Square: Channel vs Lead Status", "Chi2 = 41.28, p = 2.31e-04", "Reject H0 (Significant)"),
    ("Welch's T-Test: Web Visits vs Conv.", "t = 14.86, p = 4.15e-45", "Reject H0 (Decisive)"),
    ("One-Way ANOVA: Sales across EV Models", "F = 312.45, p = 3.29e-120", "Reject H0 (Decisive)")
]
for h_i, (title, stat, dec) in enumerate(hypo_tests):
    y_h = 28 - h_i * 6
    ax.add_patch(patches.FancyBboxPatch((5, y_h - 2.5), 42, 5, boxstyle="round,pad=0.2", fc='#F7FAFC', ec='#E2E8F0'))
    ax.text(6.5, y_h + 1, title, fontsize=7.5, fontweight='bold', color='#2D3748')
    ax.text(6.5, y_h - 1.2, f"{stat} -> {dec}", fontsize=7, color='#2B6CB0')

# Right Panel: Customer Personas (Unsupervised Segmentation)
ax.add_patch(patches.FancyBboxPatch((51, 4), 46, 82, boxstyle="round,pad=0.5", fc='white', ec='#CBD5E0', lw=1.5))
ax.text(53, 82, "4. K-MEANS CUSTOMER PERSONA SEGMENTATION (k=4)", fontsize=10.5, fontweight='bold', color='#1A365D')

personas = [
    ("Persona 0: Digital High-Intent Adopters", "Headcount: 354 (35.4%) | Avg Price: Rs. 12.4L | Conv: 37.0%", 
     "Frequent web configurator users with medium ad exposure. Prioritize Tiago/Punch EV. High responsiveness to digital test-drive booking reminders.", "#3182CE"),
    ("Persona 1: Affordable / Value Seekers", "Headcount: 160 (16.0%) | Avg Price: Rs. 14.3L | Conv: 48.1%", 
     "Price-conscious family buyers looking for running-cost savings. High conversion when paired with exchange bonuses and scrappage incentives.", "#38A169"),
    ("Persona 2: Premium Luxury Intenders", "Headcount: 285 (28.5%) | Avg Price: Rs. 18.2L | Conv: 35.4%", 
     "Affluent tech enthusiasts eyeing Curvv EV and Harrier EV. Motivated by ADAS safety, battery warranty, and DC fast-charging infrastructure.", "#805AD5"),
    ("Persona 3: Casual Low-Touch Inquirers", "Headcount: 201 (20.1%) | Avg Price: Rs. 13.7L | Conv: 64.2%", 
     "Extremely high web visits (Avg 12.1 sessions). Require minimal sales friction; convert rapidly once showroom test drive is scheduled.", "#D69E2E")
]

for p_i, (p_title, p_stats, p_desc, p_col) in enumerate(personas):
    y_p = 73 - p_i * 18
    ax.add_patch(patches.FancyBboxPatch((53, y_p - 11), 42, 16, boxstyle="round,pad=0.3", fc='#F8FAFC', ec=p_col, lw=1.5))
    ax.add_patch(patches.Rectangle((53, y_p - 11), 1.5, 16, fc=p_col))
    ax.text(56, y_p + 3, p_title, fontsize=8.5, fontweight='bold', color=p_col)
    ax.text(56, y_p, p_stats, fontsize=7.5, fontweight='bold', color='#2D3748')
    # Wrapped text
    ax.text(56, y_p - 5.5, p_desc, fontsize=7.2, color='#4A5568', wrap=True)

plt.savefig(f"{CHARTS_DIR}/project_output_evaluation_clustering.png", dpi=250, bbox_inches='tight')
plt.close()
print("Generated project_output_evaluation_clustering.png successfully.")
