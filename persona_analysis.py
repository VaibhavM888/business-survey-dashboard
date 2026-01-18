import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette

# --- STEP 1: LOAD & CLEAN DATA ---
print("Loading and cleaning data...")
file_path = 'Lifestyle Survey Result.csv'
df = pd.read_csv(file_path)

# Select the "Purchase Driver" trade-off variables (Q49 Series)
q49_cols = [f'Q49010{i}' for i in range(1, 9)]
data = df[q49_cols].copy()

# Clean Q49: Force numeric and fill missing values with median
# IMPORTANT: We update 'df' as well so we can calculate stats later!
for col in q49_cols:
    clean_col = pd.to_numeric(data[col], errors='coerce')
    data[col] = clean_col
    df[col] = clean_col 

data = data.fillna(data.median())
df[q49_cols] = df[q49_cols].fillna(data.median())

# Scale the data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- STEP 2: CLUSTERING ---
print("Running clustering...")
# Generate the Linkage Matrix using Ward's method
Z = linkage(data_scaled, method='ward')

# Create 4 Clusters
k = 4
cluster_ids = fcluster(Z, t=k, criterion='maxclust')
df['Cluster_ID'] = cluster_ids

# --- STEP 3: ASSIGN UNIQUE LABELS (RANKING METHOD) ---
# Calculate means to identify characteristics
means = df.groupby('Cluster_ID')[q49_cols].mean()
remaining_clusters = list(means.index)
cluster_labels = {}

# Logic: Find the cluster that maximizes/minimizes specific traits to ensure unique naming
# 1. Budget Pragmatists (Highest preference for Low Price - Q490103 close to 2)
budget_id = means.loc[remaining_clusters]['Q490103'].idxmax()
cluster_labels[budget_id] = "Budget Pragmatists (Price Focus)"
remaining_clusters.remove(budget_id)

# 2. Dynamic Drivers (Strongest preference for Dynamic Feel - Q490107 close to 1)
dynamic_id = means.loc[remaining_clusters]['Q490107'].idxmin()
cluster_labels[dynamic_id] = "Dynamic Drivers (Driving Feel)"
remaining_clusters.remove(dynamic_id)

# 3. Stylish Comfort (Strongest preference for Design - Q490105 close to 1)
stylish_id = means.loc[remaining_clusters]['Q490105'].idxmin()
cluster_labels[stylish_id] = "Stylish Comfort Seekers (Design)"
remaining_clusters.remove(stylish_id)

# 4. Premium Quality (The remaining group)
premium_id = remaining_clusters[0]
cluster_labels[premium_id] = "Premium Quality Seekers (Feature Focus)"

print("Labels Assigned:", cluster_labels)

# --- STEP 4: PLOT DENDROGRAM ---
plt.figure(figsize=(14, 8))
plt.title("Customer Segmentation Dendrogram\n(Customer Personas)", fontsize=16)
plt.xlabel("Respondents")
plt.ylabel("Euclidean Distance")

# Custom Palette: Red, Blue, Green, Purple
custom_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
set_link_color_palette(custom_palette)

dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.,
           color_threshold=Z[-k+1, 2], above_threshold_color='grey')

# Create Legend
handles = []
sorted_ids = sorted(cluster_labels.keys()) 
for i, cid in enumerate(sorted_ids):
    label = cluster_labels[cid]
    color = custom_palette[i % len(custom_palette)] 
    handles.append(mpatches.Patch(color=color, label=f"Group {cid}: {label}"))

plt.legend(handles=handles, title="Persona Key Characteristics", loc='upper right')
plt.tight_layout()
plt.show() # <--- Close the window to continue!

# --- STEP 5: KPI PROFILE TABLE ---
print("\nGenerating KPI Profile...")

# FIX: Explicitly clean Demographic columns (Age & Income)
demo_cols = ['SQ0502', 'DQ08']
for col in demo_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Select columns to profile
kpi_cols = demo_cols + q49_cols
cluster_profile = df.groupby('Cluster_ID')[kpi_cols].mean()
cluster_profile['Count (N)'] = df['Cluster_ID'].value_counts()

# Rename columns
rename_dict = {
    'SQ0502': 'Avg Age',
    'DQ08': 'Avg Income',
    'Q490101': 'Ext/Int Design',
    'Q490102': 'Ext/Int Color',
    'Q490103': 'Price/Option',
    'Q490104': 'New/Proven', 
    'Q490105': 'Design/Perf',
    'Q490106': 'Perf/Cost',
    'Q490107': 'Dynamic/Comfort',
    'Q490108': 'Popular/Rare'
}
cluster_profile = cluster_profile.rename(columns=rename_dict)
print(cluster_profile.round(2))

# --- STEP 6: NORMALIZED HEATMAP ---
print("\nGenerating Heatmap...")
plot_data = cluster_profile.drop(columns=['Count (N)']).T
# Z-Score Normalization
plot_data_norm = plot_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

plt.figure(figsize=(12, 8))
plt.title("Cluster DNA: Relative Strengths & Weaknesses\n(Red = Higher than Avg, Blue = Lower than Avg)", fontsize=14)
sns.heatmap(plot_data_norm, cmap='RdBu_r', center=0, annot=cluster_profile.drop(columns=['Count (N)']).T, fmt=".2f", linewidths=.5)
plt.xlabel("Customer Persona Group")
plt.tight_layout()
plt.show()
