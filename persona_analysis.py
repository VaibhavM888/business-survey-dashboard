import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette

# ==============================================================================
# 1. SETUP & MAPPINGS
# ==============================================================================
FILE_PATH = 'Lifestyle Survey Result.csv'

# Mappings for Q49 (Values) - These have high data quality (N=30k)
# Based on typical Semantic Differential scales (1 vs 7)
VARIABLE_LABELS = {
    'Q490101': 'Brand Driven',        # High = Prefers Brand over Product
    'Q490102': 'Socially Influenced', # High = Cares about others' view
    'Q490103': 'Price Sensitive',     # High = Price > Performance
    'Q490104': 'Novelty Seeker',      # High = Likes New Models
    'Q490105': 'Design Focused',      # High = Design > Performance
    'Q490106': 'Cost over Perf',      # High = Cost > Performance
    'Q490107': 'Comfort Focused',     # High = Comfort > Driving Dynamics
    'Q490108': 'Trend Follower'       # High = Popularity > Uniqueness
}

# ==============================================================================
# 2. DATA PROCESSING
# ==============================================================================
def load_and_prep_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return None, None

    # Use Q49 columns (Values) which are robust and complete
    cols = list(VARIABLE_LABELS.keys())
    
    # Clean: Coerce to numeric (fixes ' ' error) and drop rows with missing values
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Dropna: specific to these columns
    df_clean = df.dropna(subset=cols).copy()
    
    # Downsample for Dendrogram visibility if dataset is huge (optional but good for speed)
    # We'll use the full dataset for clustering but plot a sample if needed. 
    # For now, we cluster the aggregated profiles or full data? 
    # Standard approach: Cluster full data, but dendrogram might get messy if N=30k.
    # We will compute Linkage on a representative sample or the full set if memory allows.
    # Ward's linkage on 30k is heavy. We'll sample 2000 for the visualization of the tree structure.
    
    return df_clean, cols

# ==============================================================================
# 3. ANALYSIS & VISUALIZATION
# ==============================================================================
def run_cluster_analysis():
    print("--- Processing Data ---")
    df, cols = load_and_prep_data(FILE_PATH)
    if df is None: return
    
    print(f"Data Loaded: {len(df)} respondents")

    # 1. Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])

    # 2. Clustering (Ward's Method)
    # Note: For N=30k, we calculate the Z linkage on a sample for the Dendrogram, 
    # but assign clusters to everyone using K-Means or just use the sample.
    # To keep it rigorous but fast, we'll plot the dendrogram on a Sample (N=5000)
    # but derive the profiles from the full dataset using Agglomerative or K-Means.
    # Let's stick to Ward on a sample for the Visual, and K-Means (k=3) for the Profiles.
    # This aligns the visual "Tree" concept with the robust "Group" profiles.
    
    # Sample for Dendrogram
    sample_idx = np.random.choice(len(X_scaled), min(len(X_scaled), 5000), replace=False)
    X_sample = X_scaled[sample_idx]
    Z = linkage(X_sample, method='ward')

    # Assign Clusters to All Data (using K-Means to approximate the 3-cut from Ward)
    # This ensures our Heatmap covers everyone.
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)

    # 3. Create Profiles (Mean Z-Scores)
    # We use the scaled data to get Z-scores directly (Mean=0)
    df_scaled = pd.DataFrame(X_scaled, columns=cols)
    df_scaled['Cluster_ID'] = df['Cluster_ID'].values
    profile = df_scaled.groupby('Cluster_ID').mean()
    profile.rename(columns=VARIABLE_LABELS, inplace=True)

    # 4. Naming Logic (Distinctive Traits)
    persona_info = {}
    for cid in profile.index:
        # Find traits with highest positive Z-scores (Strongest adherence)
        sorted_traits = profile.loc[cid].sort_values(ascending=False)
        top_3 = sorted_traits.head(3).index.tolist()
        desc = ", ".join(top_3)
        
        # Name
        lead = top_3[0]
        if lead in ['Brand Driven', 'Design Focused', 'Novelty Seeker']:
            name = "Aspirational Trendsetter"
        elif lead in ['Price Sensitive', 'Cost over Perf', 'Socially Influenced']:
            name = "Pragmatic Value-Seeker"
        elif lead in ['Comfort Focused', 'Trend Follower']:
            name = "Passive Mainstreamer"
        else:
            name = f"Segment {cid}"
            
        persona_info[cid] = {'name': name, 'desc': desc}

    # ==========================================================================
    # VISUAL 1: DENDROGRAM WITH CLEAN LEGEND
    # ==========================================================================
    print("--- Generating Dendrogram ---")
    plt.figure(figsize=(12, 8))
    
    # Custom Palette
    custom_palette = ['#2ca02c', '#d62728', '#1f77b4'] # Green, Red, Blue
    set_link_color_palette(custom_palette)
    
    dendrogram(
        Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.,
        show_contracted=True, color_threshold=Z[-3+1, 2], above_threshold_color='grey'
    )
    
    plt.title("Psychographic Segmentation Hierarchy (Sampled N=5000)", fontsize=16)
    plt.xlabel("Cluster Size", fontsize=12)
    plt.ylabel("Dissimilarity", fontsize=12)

    # LEGEND (No 'Cluster X' text)
    patches = []
    # Match K-Means labels to colors (Approximate for visual consistency)
    sorted_ids = sorted(persona_info.keys())
    for i, cid in enumerate(sorted_ids):
        color = custom_palette[i % len(custom_palette)]
        info = persona_info[cid]
        # Format: "Name \n (Trait, Trait, Trait)"
        label = f"{info['name']}\n({info['desc']})"
        patches.append(mpatches.Patch(color=color, label=label))

    plt.legend(handles=patches, title="Persona & Key Drivers", 
               loc='upper right', fontsize=10, title_fontsize=12, shadow=True)
    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # VISUAL 2: PERFECTED HEATMAP
    # ==========================================================================
    print("--- Generating Scorecard Heatmap ---")
    
    # Prepare Data: Transpose for "Scorecard" look (Traits=Y, Personas=X)
    heatmap_data = profile.T 
    
    # Rename Columns
    new_cols = {cid: persona_info[cid]['name'] for cid in heatmap_data.columns}
    heatmap_data.rename(columns=new_cols, inplace=True)
    
    # ROW-WISE STANDARDIZATION
    # This highlights relative strengths: "Does Persona A care about this MORE than Persona B?"
    # (Value - RowMean) / RowStd
    heatmap_norm = heatmap_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    plt.figure(figsize=(8, 10))
    sns.heatmap(
        heatmap_norm, 
        cmap='RdBu_r',      # Red = High, Blue = Low
        center=0,           # White = Average
        annot=True,         # Show Z-score numbers
        fmt=".1f",          
        linewidths=.5,
        vmin=-2, vmax=2,    # FORCE CONTRAST: Caps values to ensure gradient is visible
        cbar_kws={'label': 'Relative Strength (Z-Score)'}
    )
    plt.title("Persona DNA Scorecard\n(Red = Strong Preference, Blue = Low Preference)", fontsize=14)
    plt.xlabel("Target Personas", fontsize=12)
    plt.ylabel(None)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_cluster_analysis()