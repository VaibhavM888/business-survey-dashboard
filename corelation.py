import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set plot style for academic report
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_clean_data():
    print(">> Loading Datasets...")
    
    # Load necessary files
    df_intent = pd.read_csv("Car purchase intention Survey Result.csv")
    df_sales = pd.read_csv("Sales Satisfaction Survey results.csv")
    
    # --- DATA CLEANING & FEATURE ENGINEERING ---
    
    # 1. Calculate AGE (Assuming Current Year 2025)
    current_year = 2025
    df_intent['Age'] = current_year - pd.to_numeric(df_intent['SQ0502'], errors='coerce')
    df_sales['Age'] = current_year - pd.to_numeric(df_sales['SQ0502'], errors='coerce')
    
    # 2. Clean INCOME
    df_intent['Income_Code'] = pd.to_numeric(df_intent['DQ08'], errors='coerce')
    
    # 3. Clean SATISFACTION SCORES (RQ3)
    # Map columns to readable names
    df_sales = df_sales.rename(columns={
        'Q0512': 'Overall_Sat',
        'Q0509': 'Salesperson_Score',
        'Q0406': 'Facility_Score',
        'Q0511': 'Delivery_Score'
    })

    # Convert to numeric, turning errors into NaNs
    cols_to_clean = ['Overall_Sat', 'Salesperson_Score', 'Facility_Score', 'Delivery_Score']
    for col in cols_to_clean:
        df_sales[col] = pd.to_numeric(df_sales[col], errors='coerce')

    # IMPORTANT: Drop rows where ANY of the key variables are NaN
    # This fixes the "exog contains inf or nans" error
    df_sales = df_sales.dropna(subset=cols_to_clean)

    # 4. Clean FUEL PREFERENCE (RQ1/RQ2)
    df_intent['Fuel_Type_Code'] = pd.to_numeric(df_intent['Q0119'], errors='coerce')
    # 1=Gas, 2=Diesel, 3=Hybrid, 4=EV (Check your codebook if 5/6 exist)
    # We treat 3, 4, 5 as "Eco-Friendly"
    df_intent['Is_Eco_Buyer'] = df_intent['Fuel_Type_Code'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    
    # Drop NaNs for the demographic analysis
    df_intent = df_intent.dropna(subset=['Age', 'Income_Code', 'Is_Eco_Buyer'])
    
    return df_intent, df_sales

def analyze_rq1_rq2_demographics(df):
    """
    RQ1 & RQ2: Impact of Demographics on Eco-Friendly Purchase Intent
    """
    print("\n" + "="*50)
    print(" RESEARCH QUESTION 1 & 2: DEMOGRAPHIC DRIVERS")
    print("="*50)
    
    X = df[['Age', 'Income_Code']]
    y = df['Is_Eco_Buyer']
    
    # 1. CORRELATION ANALYSIS
    print("\n[1.1] Correlation Matrix (Demographics vs Eco-Choice)")
    corr = df[['Age', 'Income_Code', 'Is_Eco_Buyer']].corr()
    print(corr)
    
    # 2. LOGISTIC REGRESSION
    print("\n[1.2] Logistic Regression Results (Predicting Eco-Adoption)")
    
    # Add constant for intercept
    X_const = sm.add_constant(X) 
    
    try:
        logit_model = sm.Logit(y, X_const).fit()
        print(logit_model.summary())
        
        print("\n>> INTERPRETATION:")
        params = logit_model.params
        pvalues = logit_model.pvalues
        
        if pvalues['Age'] > 0.05:
            print(f"   - Age (P={pvalues['Age']:.3f}): Result is NOT statistically significant (P > 0.05).")
            print("     This means Age alone does not strongly predict Eco-choice in this specific dataset.")
        else:
            print(f"   - Age (P={pvalues['Age']:.3f}): Significant predictor.")
            
    except Exception as e:
        print(f"Regression Error: {e}")

def analyze_rq3_strategy(df):
    """
    RQ3: Strategic Approach (Drivers of Satisfaction)
    """
    print("\n" + "="*50)
    print(" RESEARCH QUESTION 3: DRIVERS OF SATISFACTION")
    print("="*50)
    
    # Define variables
    X = df[['Salesperson_Score', 'Facility_Score', 'Delivery_Score']]
    y = df['Overall_Sat']
    
    # 1. CORRELATION
    print("\n[2.1] Correlation with Overall Satisfaction")
    print(df[['Overall_Sat', 'Salesperson_Score', 'Facility_Score', 'Delivery_Score']].corr()['Overall_Sat'])
    
    # 2. MULTIPLE LINEAR REGRESSION
    print("\n[2.2] Multiple Linear Regression Results (Key Drivers)")
    
    # Ensure no infinite values
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index] # Align y with X
    
    X_const = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X_const).fit()
        print(model.summary())
        
        # Feature Importance Visualization
        coefs = model.params.drop('const')
        plt.figure()
        coefs.sort_values().plot(kind='barh', color='teal')
        plt.title("Key Drivers of Customer Satisfaction (Regression Coefficients)")
        plt.xlabel("Impact on Satisfaction (Coefficient)")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Regression Error: {e}")

if __name__ == "__main__":
    df_intent, df_sales = load_and_clean_data()
    
    analyze_rq1_rq2_demographics(df_intent)
    analyze_rq3_strategy(df_sales)