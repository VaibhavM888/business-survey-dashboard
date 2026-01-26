import pandas as pd

import numpy as np

import warnings

from sklearn.tree import DecisionTreeClassifier, export_text



# 1. ELIMINATE WARNINGS (Clean Output)

warnings.filterwarnings("ignore")



# ==========================================

# SETUP & DATA LOADING

# ==========================================

FILE_NAME = "Car purchase intention Survey Result.csv"



def load_and_clean_data():

    print(">> Loading and cleaning data...")

    try:

        df = pd.read_csv(FILE_NAME)

    except FileNotFoundError:

        print(f"Error: {FILE_NAME} not found. Please check directory.")

        return None



    # Clean numeric columns

    for col in ['SQ0502', 'DQ08', 'Q1903', 'Q1907']:

        df[col] = pd.to_numeric(df[col], errors='coerce')



    # Age Logic

    df['Age'] = np.where(df['SQ0502'] < 1900, df['SQ0502'], 2024 - df['SQ0502'])

   

    return df



# ==========================================

# MODULE 1: STRATEGIC SALES PERSONA

# ==========================================

def run_sales_persona_module(df):

    print("\n" + "="*95)

    print("MODULE 1: STRATEGIC SALES PERSONA MODELING")

    print("Goal: Prescribe the optimal 'Sales Pitch' and 'Tactical Action' by demographic.")

    print("="*95)



    # 1. Define Professional Strategy Labels

    def classify_strategy(code):

        if code in [1, 2, 3, 15, 16, 17, 27, 29]: return 'SELL IMAGE (Brand/Design)'

        elif code in [7, 13, 14, 19, 21, 22, 24, 26]: return 'SELL VALUE (Price/Practicality)'

        elif code in [8, 9, 10, 11, 12, 18, 25]: return 'SELL SPECS (Performance/Tech)'

        else: return 'Other'



    # 2. Add Tactical Advice Column

    def get_tactical_advice(strategy_label):

        if "IMAGE" in strategy_label: return "Showcase Awards & Aesthetics"

        elif "VALUE" in strategy_label: return "Focus on Financing & Fuel Econ"

        elif "SPECS" in strategy_label: return "Demo Infotainment & Engine"

        else: return "-"



    df['Strategy_Label'] = df['Q1903'].apply(classify_strategy)

    model_data = df[df['Strategy_Label'] != 'Other'].dropna(subset=['Age', 'DQ08'])



    # 3. Train Tree

    X = model_data[['Age', 'DQ08']]

    y = model_data['Strategy_Label']

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)

    clf.fit(X, y)



    # 4. Simulation Table

    print("\n>> SIMULATION RESULTS (Copy into 'Recommendations' section):")

   

    personas = [

        {'Age': 24, 'DQ08': 2, 'Desc': 'Gen Z / Low Income'},

        {'Age': 28, 'DQ08': 9, 'Desc': 'Gen Z / High Income'},

        {'Age': 45, 'DQ08': 6, 'Desc': 'Millennial / Mid Income'},

        {'Age': 65, 'DQ08': 12, 'Desc': 'Boomer / High Income'},

    ]

   

    # Formatted Header

    print(f"{'CUSTOMER PROFILE':<25} | {'RECOMMENDED STRATEGY':<32} | {'TACTICAL ACTION'}")

    print("-" * 95)

   

    for p in personas:

        input_df = pd.DataFrame([[p['Age'], p['DQ08']]], columns=['Age', 'DQ08'])

        pred = clf.predict(input_df)[0]

        advice = get_tactical_advice(pred)

        print(f"{p['Desc']:<25} | {pred:<32} | {advice}")



    # 5. Rules Appendix

    print("\n>> GENERATED LOGIC RULES (For Appendix):")

    print(export_text(clf, feature_names=['Age', 'Income']))





# ==========================================

# MODULE 2: GREEN CONVERSION ENGINE

# ==========================================

def run_green_conversion_module(df):

    print("\n" + "="*95)

    print("MODULE 2: GREEN CONVERSION STRATEGY")

    print("Goal: Identify 'Rational Triggers' and prescribe 'Marketing Tactics' for EV Adoption.")

    print("="*95)



    # Full Label Map

    FULL_REASON_MAP = {

        1: "Good Brand Exp", 2: "Company/Brand", 3: "Personal Connection",

        4: "Advertising/PR", 5: "After-sales Svc", 6: "Salesperson Rec",

        7: "Short Wait Time", 8: "Engine/Power", 9: "Safety",

        10: "Adv. Features", 11: "Spacious Seats", 12: "Riding Comfort",

        13: "Trunk Space", 14: "Default Options", 15: "Exterior Style",

        16: "Interior Design", 17: "Color", 18: "Size (Length)",

        19: "Price/Discount", 20: "Tax Benefits", 21: "Resale Value",

        22: "Fuel Economy", 23: "Fuel Type", 24: "Product Quality",

        25: "Quietness", 26: "Durability", 27: "Reputation",

        28: "New/Latest Model", 29: "Eco-Friendly Image", 30: "High Usability"

    }



    # RECOMMENDATION LOGIC ENGINE

    def get_green_tactic(reason_code):

        if reason_code == 20: return "Highlight Gov. Subsidies & Tax Breaks"

        elif reason_code == 21: return "Showcase 5-Year Resale Value Charts"

        elif reason_code == 29: return "Emphasize Carbon Footprint Reduction"

        elif reason_code == 8:  return "Demo Torque & Acceleration (0-60)"

        elif reason_code == 30: return "Feature 'Camping Mode' & Utility"

        elif reason_code == 15: return "STOP: Pivot to Performance Specs"

        elif reason_code == 19: return "STOP: Pivot to Total Cost of Ownership"

        else: return "General EV Awareness"



    eco_df = df[df['Q1907'].isin([1, 2, 3, 4, 5, 6])].copy()

    eco_df['Is_Eco'] = eco_df['Q1907'].apply(lambda x: 1 if x in [4, 5, 6] else 0)



    # Calculate Conversion

    conversion = eco_df.groupby('Q1903')['Is_Eco'].agg(['mean', 'count']).reset_index()

    conversion.columns = ['Reason_Code', 'Rate', 'Count']

    robust = conversion[conversion['Count'] > 10].sort_values(by='Rate', ascending=False)



    print("\n>> TOP 5 'GREEN TRIGGERS' (Build EV marketing around these):")

    # New Layout: Added Tactic Column

    print(f"{'PRIMARY MOTIVATION (Label)':<30} | {'CONV. RATE':<10} | {'RECOMMENDED TACTIC'}")

    print("-" * 95)

   

    for _, row in robust.head(5).iterrows():

        code = int(row['Reason_Code'])

        label = FULL_REASON_MAP.get(code, "Feature")

        tactic = get_green_tactic(code)

        display_label = f"{label} (Code {code})"

        print(f"{display_label:<30} | {row['Rate']:.1%}     | {tactic}")



    print("\n>> TOP 5 'BARRIERS' (Strategic Pivot Required):")

    print(f"{'PRIMARY MOTIVATION (Label)':<30} | {'CONV. RATE':<10} | {'RECOMMENDED TACTIC'}")

    print("-" * 95)

   

    for _, row in robust.tail(5).sort_values(by='Rate').iterrows():

        code = int(row['Reason_Code'])

        label = FULL_REASON_MAP.get(code, "Feature")

        tactic = get_green_tactic(code)

        display_label = f"{label} (Code {code})"

        print(f"{display_label:<30} | {row['Rate']:.1%}     | {tactic}")



if __name__ == "__main__":

    dataset = load_and_clean_data()

    if dataset is not None:

        run_sales_persona_module(dataset)

        run_green_conversion_module(dataset)

    print("\n>> ANALYSIS COMPLETE.")
    