import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="Survey Analytics Dashboard", layout="wide")

# List of your specific files
SURVEY_FILES = [
    "Car purchase intention Survey Result.csv",
    "Initial Quality Evaluation Result.csv",
    "Lifestyle Survey Result.csv",
    "Overall Customer Satisfaction Survey Result.csv",
    "Sales Satisfaction Survey results.csv",
    "Vehicle Ownership Survey Results.csv",
    "Vehicle Purchasing Process Survey Result.csv"
]
CODEBOOK_FILE = "Survey Codebook.xlsx"

# --- Data Loading Functions ---
@st.cache_data
def load_codebook(filepath):
    """Loads the codebook to map variable names (e.g., SQ04) to descriptions."""
    try:
        df = pd.read_excel(filepath)
        # Create a dictionary: {'SQ04': 'Gender', ...}
        # Adjust column names 'Variable' and 'Variable Information' based on your CSV structure
        return pd.Series(df['Variable Information'].values, index=df['Variable']).to_dict()
    except Exception as e:
        st.warning(f"Could not load codebook: {e}")
        return {}

def load_data(filepath):
    """Loads a survey CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None

# --- Main Dashboard ---
st.title("📊 Descriptive Analytics Dashboard")

# Load Codebook Mapping
variable_map = load_codebook(CODEBOOK_FILE)

# Sidebar: File Selector
selected_file = st.sidebar.selectbox("Select Survey Dataset", SURVEY_FILES)

if selected_file:
    df = load_data(selected_file)
    
    if df is not None:
        st.header(f"Dataset: {selected_file}")
        
        # 1. High-Level Overview
        st.subheader("1. Data Overview")
        col1, col2 = st.columns(2)
        col1.metric("Total Respondents", df.shape[0])
        col1.metric("Total Variables", df.shape[1])
        with col2:
            st.dataframe(df.head(), height=150)

        # 2. Variable Inspector
        st.subheader("2. Variable Statistics")
        selected_col = st.selectbox("Select a Variable to Analyze", df.columns)

        if selected_col:
            # Get description from Codebook if available
            description = variable_map.get(selected_col, "No description available in Codebook")
            st.info(f"**Variable Information:** {description}")

            col_a, col_b = st.columns([1, 2])

            # Determine if data is likely Numeric or Categorical
            # Heuristic: If there are fewer than 20 unique values, treat as Categorical (Factors)
            unique_count = df[selected_col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[selected_col]) and unique_count > 20

            with col_a:
                st.markdown("### Key Statistics")
                if is_numeric:
                    # Show Mean, Median, Std Dev
                    stats = df[selected_col].describe()
                    st.table(stats)
                else:
                    # Show Frequencies
                    counts = df[selected_col].value_counts().reset_index()
                    counts.columns = [selected_col, 'Count']
                    counts['Percentage'] = (counts['Count'] / len(df)) * 100
                    st.dataframe(counts)

            with col_b:
                st.markdown("### Visualization")
                if is_numeric:
                    fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}", nbins=30)
                else:
                    fig = px.bar(
                        df[selected_col].value_counts().reset_index(), 
                        x=selected_col, 
                        y='count', 
                        title=f"Frequency of {selected_col}",
                        labels={'count': 'Count', selected_col: selected_col}
                    )
                st.plotly_chart(fig, use_container_width=True)