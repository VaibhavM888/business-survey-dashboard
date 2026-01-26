import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- LOAD DATA ---
df_life = pd.read_csv('Lifestyle Survey Result.csv')
df_process = pd.read_csv('Vehicle Purchasing Process Survey Result.csv')

# --- HELPER: CLEAN & TAG GENERATIONS ---
def prep_data(df):
    # Fix " " errors by forcing numeric
    df['SQ0502'] = pd.to_numeric(df['SQ0502'], errors='coerce')
    
    conditions = [
        (df['SQ0502'] <= 29),
        (df['SQ0502'] <= 45),
        (df['SQ0502'] <= 60),
        (df['SQ0502'] > 60)
    ]
    choices = ['Gen Z', 'Millennials', 'Gen X', 'Boomers']
    df['Gen'] = np.select(conditions, choices, default=None)
    return df.dropna(subset=['Gen'])

df_life = prep_data(df_life)
df_process = prep_data(df_process)

# --- FEATURE ENGINEERING (SCORING) ---
# Clean columns first
cols_q49 = ['Q490103', 'Q490105', 'Q490104', 'Q490106', 'Q490107']
for c in cols_q49:
    df_life[c] = pd.to_numeric(df_life[c], errors='coerce')

# Calculate Scores (1.0 = Focus, 0.0 = Ignore)
# Q490103 (1=Option, 2=Price) -> Price Focus = (Val - 1) -> 2 becomes 1
df_life['Price Focus'] = df_life['Q490103'] - 1 

# Q490105 (1=Design, 2=Perf) -> Design Focus = (2 - Val) -> 1 becomes 1
df_life['Design Focus'] = 2 - df_life['Q490105']

# Q490104 (1=New, 2=Proven) -> Tech Focus
df_life['Tech Focus'] = 2 - df_life['Q490104']

# Q490106 (1=Perf, 2=Cost) -> Performance Focus
df_life['Performance Focus'] = 2 - df_life['Q490106']

# Q490107 (1=Dynamic, 2=Comfort) -> Driving Feel
df_life['Driving Feel'] = 2 - df_life['Q490107']

# Process Scores
df_process['Q0302'] = pd.to_numeric(df_process['Q0302'], errors='coerce')
df_process['Digital Research'] = np.where(df_process['Q0302'].isin([1, 2, 3, 5]), 1, 0)

# Aggregation
profile_life = df_life.groupby('Gen')[['Price Focus', 'Design Focus', 'Tech Focus', 'Performance Focus', 'Driving Feel']].mean()
profile_proc = df_process.groupby('Gen')[['Digital Research']].mean()
final_profile = pd.concat([profile_life, profile_proc], axis=1)

# --- PLOTTING ---
fig = go.Figure()
categories = final_profile.columns.tolist()

colors = {'Gen Z': '#00ff00', 'Millennials': '#ff00ff', 'Gen X': '#0000ff', 'Boomers': '#ff0000'}

for gen in ['Gen Z', 'Millennials', 'Gen X', 'Boomers']:
    if gen in final_profile.index:
        fig.add_trace(go.Scatterpolar(
            r=final_profile.loc[gen].values,
            theta=categories,
            fill='toself',
            name=gen,
            line_color=colors[gen],
            opacity=0.6
        ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 0.6]), # Scaled to fit data
    ),
    title="Generational Purchase DNA (Radar Chart)",
    showlegend=True
)
fig.show()