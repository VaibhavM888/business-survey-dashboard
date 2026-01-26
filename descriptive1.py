import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# SECTION 1: CONFIGURATION & MAPPING
# ==============================================================================
class Config:
    FILES = {
        'lifestyle': 'Lifestyle Survey Result.csv',
        'intent': 'Car purchase intention Survey Result.csv',
        'process': 'Vehicle Purchasing Process Survey Result.csv'
    }

    CHANNEL_MAP = {
        1: 'Offline (Showroom/Salesperson)',
        2: 'Official Online Site',
        3: 'Online Community/Specialty Site'
    }

    # NEW: Grouped Category Logic
    DRIVER_CATEGORY_MAP = {
        # Economic & Value
        14: 'Economic & Value', 19: 'Economic & Value', 20: 'Economic & Value', 
        21: 'Economic & Value', 22: 'Economic & Value',
        # Brand & Trust
        1: 'Brand & Trust', 2: 'Brand & Trust', 3: 'Brand & Trust', 
        4: 'Brand & Trust', 5: 'Brand & Trust', 6: 'Brand & Trust', 27: 'Brand & Trust',
        # Design & Style
        15: 'Design & Style', 16: 'Design & Style', 17: 'Design & Style', 
        28: 'Design & Style',
        # Performance & Quality
        8: 'Performance & Quality', 9: 'Performance & Quality', 
        23: 'Performance & Quality', 24: 'Performance & Quality', 
        25: 'Performance & Quality', 26: 'Performance & Quality', 
        29: 'Performance & Quality',
        # Comfort & Utility
        10: 'Comfort & Utility', 11: 'Comfort & Utility', 12: 'Comfort & Utility', 
        13: 'Comfort & Utility', 18: 'Comfort & Utility', 30: 'Comfort & Utility',
        # Purchase Process
        7: 'Purchase Process (Fast)',
        # Other
        31: 'Other', 97: 'Other'
    }

# ==============================================================================
# SECTION 2: DATA HELPERS
# ==============================================================================
def load_data(key):
    path = Config.FILES.get(key)
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    print(f"Warning: {path} not found.")
    return None

def get_generation(age):
    try:
        age = float(age)
        if pd.isna(age): return "Unknown"
        if age <= 29: return 'Gen Z (<=29)'
        elif age <= 45: return 'Millennials (30-45)'
        elif age <= 60: return 'Gen X (46-60)'
        else: return 'Boomers (60+)'
    except:
        return "Unknown"

def save_clean_crosstab(df, row_col, col_col, filename):
    """Generates a rounded, column-normalized crosstab."""
    ct = pd.crosstab(df[row_col], df[col_col], normalize='columns') * 100
    ct_rounded = ct.round(0).astype(int)
    
    # Sort columns if possible
    desired_order = ['Gen Z (<=29)', 'Millennials (30-45)', 'Gen X (46-60)', 'Boomers (60+)']
    existing_cols = [c for c in desired_order if c in ct_rounded.columns]
    if existing_cols:
        ct_rounded = ct_rounded[existing_cols]
    
    # Sort Rows by Gen Z importance if available
    if 'Gen Z (<=29)' in ct_rounded.columns:
        ct_rounded = ct_rounded.sort_values('Gen Z (<=29)', ascending=False)
        
    ct_rounded.to_csv(filename)
    print(f"Saved: {filename}")
    print(ct_rounded)
    print("-" * 30)

# ==============================================================================
# SECTION 3: ANALYTICS MODULES
# ==============================================================================

def analyze_grouped_drivers():
    print("\n--- [RQ2] Analyzing Grouped Purchase Drivers ---")
    df = load_data('intent')
    if df is None: return

    if 'Q1903' in df.columns and 'SQ0502' in df.columns:
        df['Generation'] = df['SQ0502'].apply(get_generation)
        # Convert to numeric to ensure mapping works
        df['Driver_Code'] = pd.to_numeric(df['Q1903'], errors='coerce')
        # Apply the new Grouping Map
        df['Driver_Category'] = df['Driver_Code'].map(Config.DRIVER_CATEGORY_MAP).fillna('Other')
        
        save_clean_crosstab(df, 'Driver_Category', 'Generation', "RQ2_Grouped_Purchase_Drivers.csv")

def analyze_psychographic_segments():
    print("\n--- [RQ1] Running Psychographic Segmentation (K-Means) ---")
    df = load_data('lifestyle')
    if df is None: return

    lifestyle_cols = [f'Q48010{i}' for i in range(1, 9)] 
    
    for col in lifestyle_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=lifestyle_cols).copy()
    if df_clean.empty: return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[lifestyle_cols])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=lifestyle_cols)
    
    cluster_map = {}
    for i, row in centers.iterrows():
        if row['Q480101'] > 6.0: 
            cluster_map[i] = 'Quality-First Trend Follower'
        elif row['Q480102'] > 6.0: 
            cluster_map[i] = 'Practical Early Adopter'
        else: 
            cluster_map[i] = 'Informed Aesthete'
            
    df_clean['Segment_Label'] = df_clean['Cluster_ID'].map(cluster_map).fillna('Other')

    if 'SQ0502' in df_clean.columns:
        df_clean['Generation'] = df_clean['SQ0502'].apply(get_generation)
        save_clean_crosstab(df_clean, 'Segment_Label', 'Generation', "RQ1_Psychographic_Segments.csv")

def analyze_sales_channels():
    print("\n--- [RQ3] Analyzing Sales Channels ---")
    df = load_data('process')
    if df is None: return

    if 'Q030302' in df.columns and 'SQ0502' in df.columns:
        df['Generation'] = df['SQ0502'].apply(get_generation)
        
        df['Channel_Code'] = pd.to_numeric(df['Q030302'], errors='coerce')
        df['Channel_Label'] = df['Channel_Code'].map(Config.CHANNEL_MAP).fillna('Unknown')
        
        save_clean_crosstab(df, 'Channel_Label', 'Generation', "RQ3_Sales_Channels.csv")

if __name__ == "__main__":
    analyze_psychographic_segments()
    analyze_grouped_drivers()
    analyze_sales_channels()
    print("\nProcessing Complete.")