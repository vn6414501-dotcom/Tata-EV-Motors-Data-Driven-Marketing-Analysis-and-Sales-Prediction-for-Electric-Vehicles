"""
Tata Motors EV Marketing Analysis - Academic Synthetic Dataset Generator
Generates a realistic, statistically coherent dataset of 1,000 EV prospective customers
for academic capstone project analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_tata_ev_data(num_records=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Customer_ID
    customer_ids = [f"TATACUST{i}" for i in range(1001, 1001 + num_records)]
    
    # 2. Age (21 to 68, bell-curved around 37)
    raw_age = np.random.normal(loc=37, scale=9.5, size=num_records)
    ages = np.clip(raw_age, 21, 68).astype(int)
    
    # 3. Gender
    genders = np.random.choice(['Male', 'Female', 'Other'], size=num_records, p=[0.58, 0.39, 0.03])
    
    # 4. City (Tier 1 & Tier 2 EV hubs in India)
    cities_list = [
        'Mumbai', 'Delhi', 'Bengaluru', 'Pune', 'Hyderabad', 
        'Chennai', 'Ahmedabad', 'Kolkata', 'Jaipur', 'Kochi'
    ]
    city_weights = [0.18, 0.17, 0.16, 0.12, 0.11, 0.09, 0.06, 0.04, 0.04, 0.03]
    cities = np.random.choice(cities_list, size=num_records, p=city_weights)
    
    # 5. EV_Model & 6. Purchase_Price
    models = ['Tiago EV', 'Punch EV', 'Tigor EV', 'Nexon EV', 'Curvv EV', 'Harrier EV']
    model_weights = [0.22, 0.28, 0.08, 0.25, 0.11, 0.06]
    selected_models = np.random.choice(models, size=num_records, p=model_weights)
    
    # Realistic base price ranges (in INR)
    price_ranges = {
        'Tiago EV': (799000, 1149000),
        'Punch EV': (1099000, 1549000),
        'Tigor EV': (1249000, 1375000),
        'Nexon EV': (1449000, 1949000),
        'Curvv EV': (1749000, 2199000),
        'Harrier EV': (2199000, 2550000)
    }
    
    purchase_prices = []
    for model in selected_models:
        low, high = price_ranges[model]
        # Random price within model trim range, rounded to nearest 5,000
        p = np.random.uniform(low, high)
        purchase_prices.append(round(p / 5000) * 5000)
        
    # 7. Purchase_Date (Dates between Jan 1, 2023 and Dec 31, 2024)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_delta = (end_date - start_date).days
    random_days = np.random.randint(0, date_delta, size=num_records)
    purchase_dates = [(start_date + timedelta(days=int(d))).strftime('%Y-%m-%d') for d in random_days]
    
    # 8. Marketing_Channel
    channels = ['Social Media', 'Google Search', 'TV Commercial', 'Dealership Visit', 'Print Media', 'Email Campaign']
    channel_weights = [0.28, 0.24, 0.16, 0.14, 0.10, 0.08]
    marketing_channels = np.random.choice(channels, size=num_records, p=channel_weights)
    
    # 9. Ad_Exposure (1 to 25 exposures)
    ad_exposures = np.random.geometric(p=0.15, size=num_records) + np.random.randint(1, 5, size=num_records)
    ad_exposures = np.clip(ad_exposures, 1, 25)
    
    # 10. Campaign_Type
    campaigns = ['Festive Offer', 'Green Mobility Drive', 'Exchange Bonus', 'Referral Program', 'New Launch Blitz']
    campaign_weights = [0.30, 0.25, 0.18, 0.15, 0.12]
    campaign_types = np.random.choice(campaigns, size=num_records, p=campaign_weights)
    
    # 11. Website_Visits (1 to 30)
    website_visits = np.random.negative_binomial(n=4, p=0.4, size=num_records) + 1
    website_visits = np.clip(website_visits, 1, 30)
    
    # 12. Test_Drive (Influenced by website visits & channel)
    test_drives = []
    for i in range(num_records):
        td_prob = 0.25
        if website_visits[i] > 8:
            td_prob += 0.25
        elif website_visits[i] > 4:
            td_prob += 0.15
        if marketing_channels[i] in ['Dealership Visit', 'Google Search']:
            td_prob += 0.20
        td_prob = min(td_prob, 0.85)
        test_drives.append('Yes' if np.random.rand() < td_prob else 'No')
        
    # 13. Customer_Interest (Low, Medium, High)
    customer_interests = []
    for i in range(num_records):
        score = (website_visits[i] * 0.4) + (ad_exposures[i] * 0.2) + (3.0 if test_drives[i] == 'Yes' else 0.0)
        if score > 7.5:
            customer_interests.append('High')
        elif score > 4.0:
            customer_interests.append('Medium')
        else:
            customer_interests.append('Low')
            
    # 14. Lead_Status & 15. Sales_Amount
    lead_statuses = []
    sales_amounts = []
    
    for i in range(num_records):
        # Conversion probability formulation
        prob = 0.10
        if test_drives[i] == 'Yes':
            prob += 0.38
        if customer_interests[i] == 'High':
            prob += 0.22
        elif customer_interests[i] == 'Medium':
            prob += 0.08
        else:
            prob -= 0.06
            
        if website_visits[i] >= 10:
            prob += 0.12
        elif website_visits[i] >= 5:
            prob += 0.05
            
        if marketing_channels[i] == 'Dealership Visit':
            prob += 0.10
        elif marketing_channels[i] == 'Google Search':
            prob += 0.05
            
        if campaign_types[i] in ['Festive Offer', 'Referral Program']:
            prob += 0.05
            
        prob = float(np.clip(prob, 0.03, 0.92))
        
        is_converted = (np.random.rand() < prob)
        
        if is_converted:
            lead_statuses.append('Converted')
            # Realistic accessories, insurance addition, minor festival discount
            extra_costs = np.random.uniform(25000, 65000)
            discount = np.random.uniform(10000, 35000)
            actual_sale = purchase_prices[i] + extra_costs - discount
            sales_amounts.append(round(actual_sale, 2))
        else:
            # Unconverted distribution
            unconverted_state = np.random.choice(['Lost', 'Contacted', 'New'], p=[0.45, 0.35, 0.20])
            lead_statuses.append(unconverted_state)
            sales_amounts.append(0.0)
            
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'City': cities,
        'EV_Model': selected_models,
        'Purchase_Price': purchase_prices,
        'Purchase_Date': purchase_dates,
        'Marketing_Channel': marketing_channels,
        'Ad_Exposure': ad_exposures,
        'Campaign_Type': campaign_types,
        'Website_Visits': website_visits,
        'Test_Drive': test_drives,
        'Customer_Interest': customer_interests,
        'Lead_Status': lead_statuses,
        'Sales_Amount': sales_amounts
    })
    
    return df

if __name__ == '__main__':
    data_path = 'data/tata_ev_marketing.csv'
    df = generate_tata_ev_data(num_records=1000, random_seed=42)
    df.to_csv(data_path, index=False)
    print(f"Successfully generated {df.shape[0]} rows and {df.shape[1]} columns at {data_path}")
    print("\nFirst 5 records:")
    print(df.head())
    print("\nLead Status breakdown:")
    print(df['Lead_Status'].value_counts())
    print(f"Overall Conversion Rate: {(df['Lead_Status'] == 'Converted').mean():.2%}")
    print("\nEV Model breakdown:")
    print(df['EV_Model'].value_counts())
