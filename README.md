# Tata Motors EV Marketing Analysis Using Advanced Data Science and Machine Learning

**Academic Level:** MCA Semester-End Capstone Project  
**Course:** Advanced Data Science & Machine Learning  
**Target Enterprise:** Tata Motors Passenger Electric Vehicles (TPEM)  
**Author:** MCA Candidate  

---

## 📌 Project Overview
This capstone project implements an academic, enterprise-grade data science framework to analyze customer acquisition, marketing touchpoint effectiveness, sales revenue, and lead conversion dynamics for Tata Motors' Electric Vehicle portfolio (Tiago EV, Punch EV, Tigor EV, Nexon EV, Curvv EV, and Harrier EV).

The project adheres to the standard **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology and demonstrates core competencies in data engineering, exploratory data analysis, publication-quality visualization, inferential statistical testing, dual-track machine learning (Regression & Classification), unsupervised clustering, and interactive web dashboard development.

---

## 🏛️ Project Directory Structure

```text
Tata_Motors_EV_Marketing_Analysis/
├── data/
│   ├── tata_ev_marketing.csv              # Primary academic dataset (1,000 records × 15 features)
│   └── tata_ev_marketing_enriched.csv     # Enriched dataset with engineered features & clusters
├── notebooks/
│   └── Tata_EV_Marketing_Analysis.ipynb   # Complete 24-section interactive Jupyter Notebook
├── models/
│   ├── sales_model.pkl                    # Best trained Regressor pipeline bundle
│   └── conversion_model.pkl               # Best trained Classifier pipeline bundle
├── app/
│   └── app.py                             # Interactive Streamlit Web Application
├── scripts/
│   ├── generate_data.py                   # Reproducible dataset generator script
│   ├── ml_engine.py                       # Transparent pure-Python/NumPy ML engine
│   ├── run_pipeline.py                    # End-to-end analytical & ML pipeline
│   └── build_notebook.py                  # Jupyter Notebook generation script
├── outputs/
│   ├── charts/                            # 14 publication-quality visualization charts (.png)
│   └── results/                           # Evaluation metrics, stats JSON, and correlation CSVs
├── requirements.txt                       # Project dependencies
└── README.md                              # Complete project documentation
```

---

## 📊 Dataset Schema (15 Features)

| No. | Feature Name | Data Type | Description |
| :---: | :--- | :---: | :--- |
| 1 | `Customer_ID` | String | Unique prospect identifier (`TATACUST1001` - `TATACUST2000`) |
| 2 | `Age` | Integer | Prospective customer age (21 to 68) |
| 3 | `Gender` | Categorical | Male, Female, Other |
| 4 | `City` | Categorical | Major hubs: Mumbai, Delhi, Bengaluru, Pune, Hyderabad, etc. |
| 5 | `EV_Model` | Categorical | Tiago EV, Punch EV, Tigor EV, Nexon EV, Curvv EV, Harrier EV |
| 6 | `Purchase_Price` | Float (INR) | Baseline vehicle price (₹7.99 Lakh to ₹25.50 Lakh) |
| 7 | `Purchase_Date` | Datetime | Inquiry date (Jan 2023 - Dec 2024) |
| 8 | `Marketing_Channel` | Categorical | Social Media, Google Search, TV, Dealership Visit, Print, Email |
| 9 | `Ad_Exposure` | Integer | Number of advertisement exposures (1 to 25) |
| 10 | `Campaign_Type` | Categorical | Festive Offer, Green Mobility Drive, Exchange Bonus, etc. |
| 11 | `Website_Visits` | Integer | Digital website & configurator sessions (1 to 30) |
| 12 | `Test_Drive` | Categorical | Showroom or home test drive completed (`Yes`, `No`) |
| 13 | `Customer_Interest` | Categorical | CRM qualification tier (`Low`, `Medium`, `High`) |
| 14 | `Lead_Status` | Categorical | Funnel state (`New`, `Contacted`, `Converted`, `Lost`) |
| 15 | `Sales_Amount` | Float (INR) | Realized invoice revenue (₹0 if unconverted) |

---

## ⚙️ Machine Learning Architecture

### 1. Track A: Sales Revenue Prediction (Regression)
- **Target Variable:** `Sales_Amount` (Continuous INR)
- **Algorithms Implemented:**
  - Linear Regression (Ordinary Least Squares with L2 regularization)
  - Random Forest Regressor (Ensemble Bootstrap CART with MSE variance reduction)
  - Gradient Boosting Regressor (Sequential Residual Shrinkage)
- **Evaluation Metrics:** MAE, MSE, RMSE, $R^2$ Score

### 2. Track B: Lead Conversion Propensity (Classification)
- **Target Variable:** `Conversion_Flag` ($1 = \text{Converted}, 0 = \text{Unconverted}$)
- **Algorithms Implemented:**
  - Logistic Regression (Binary Cross-Entropy Gradient Descent)
  - Random Forest Classifier (Bootstrap CART with Gini Impurity)
  - Gradient Boosting Classifier (Log-odds Pseudo-Residual Boosting)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 3. Track C: Unsupervised Customer Segmentation (Clustering)
- **Algorithm:** K-Means Clustering ($k=4$)
- **Identified Personas:**
  - *Cluster 0:* **Digital High-Intent Adopters**
  - *Cluster 1:* **Affordable/Value Seekers**
  - *Cluster 2:* **Premium Luxury Intenders**
  - *Cluster 3:* **Casual Low-Touch Inquirers**

---

## 📈 Key Model Performance Summary

### Regression Benchmark (Sales Prediction)
| Model | MAE (INR) | RMSE (INR) | R² Score |
| :--- | :---: | :---: | :---: |
| Linear Regression | ₹5,04,397 | ₹6,43,606 | 0.3346 |
| **Random Forest Regressor** | **₹4,99,460** | **₹6,26,780** | **0.3690** |
| Gradient Boosting Regressor | ₹4,81,022 | ₹6,31,785 | 0.3588 |

### Classification Benchmark (Conversion Propensity)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **76.0%** | **72.4%** | **72.4%** | **0.7241** | **0.8318** |
| Random Forest Classifier | 75.0% | 71.3% | 71.3% | 0.7126 | 0.8271 |
| Gradient Boosting Classifier | 76.0% | 73.5% | 70.1% | 0.7176 | 0.8208 |

---

## 🚀 How to Run the Project

### 1. Setup Environment
```bash
python -m pip install -r requirements.txt
```

### 2. Run Data Generation & ML Pipeline
```bash
# Generate synthetic dataset
python scripts/generate_data.py

# Run complete analytics, statistical testing, and ML pipeline
python scripts/run_pipeline.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```

### 4. Open Jupyter Notebook
```bash
jupyter notebook notebooks/Tata_EV_Marketing_Analysis.ipynb
```

---

## 🎓 Academic Notice
*This capstone project uses a synthetic academic dataset engineered for educational and analytical purposes. It does not represent proprietary or confidential Tata Motors customer records.*
