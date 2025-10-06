import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crime Analysis Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample dataset links
SAMPLE_DATASETS = {
    "South Africa Crime Data": {
        "crime": "https://raw.githubusercontent.com/plotly/datasets/master/south_africa_crime.csv",
        "population": "https://raw.githubusercontent.com/plotly/datasets/master/south_africa_population.csv"
    },
    "Sample Crime Data 1": {
        "crime": "https://raw.githubusercontent.com/datasets/crime-data/master/data/crime.csv",
        "population": "https://raw.githubusercontent.com/datasets/population/master/data/population.csv"
    }
}

# Title and description
st.title("🚨 Crime Hotspot Analysis Dashboard")
st.markdown("""
This dashboard analyzes crime data to identify hotspots, predict high-risk areas, 
and forecast crime trends using machine learning and statistical models.
""")

# Sidebar for data source selection
st.sidebar.header("📁 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload your own files", "Use sample datasets"]
)

# Initialize data variables
crime_data = None
population_data = None

if data_source == "Upload your own files":
    st.sidebar.header("File Upload")
    uploaded_crime = st.sidebar.file_uploader("Upload Crime Data CSV", type=["csv"], key="crime")
    uploaded_pop = st.sidebar.file_uploader("Upload Population Data CSV", type=["csv"], key="pop")
    
    crime_data = uploaded_crime
    population_data = uploaded_pop

else:
    st.sidebar.header("Sample Datasets")
    selected_dataset = st.sidebar.selectbox(
        "Choose sample dataset:",
        list(SAMPLE_DATASETS.keys())
    )
    
    dataset_info = SAMPLE_DATASETS[selected_dataset]
    
    st.sidebar.markdown(f"""
    **Selected Dataset:** {selected_dataset}
    
    **Crime Data URL:** [{dataset_info['crime'].split('/')[-1]}]({dataset_info['crime']})
    
    **Population Data URL:** [{dataset_info['population'].split('/')[-1]}]({dataset_info['population']})
    """)
    
    # Load sample data
    try:
        crime_data = dataset_info['crime']
        population_data = dataset_info['population']
        st.sidebar.success("✅ Sample dataset selected!")
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {str(e)}")

# Configuration parameters
st.sidebar.header("⚙️ Analysis Parameters")
percentile_threshold = st.sidebar.slider("Hotspot Percentile Threshold", 0.5, 0.95, 0.8, 0.05)
n_forecast = st.sidebar.slider("Forecast Periods", 1, 36, 24)

# Quick start with demo data
st.sidebar.header("🚀 Quick Start")
if st.sidebar.button("Use Demo Data"):
    st.sidebar.info("Using built-in demo data for immediate analysis")

# Load utility functions (same as before)
def compute_station_features(df):
    years = np.asarray(df['YearStart'].values, dtype=float)
    counts = np.asarray(df['Count'].values, dtype=float)

    if len(counts) == 0:
        mean = 0.0
        std = 0.0
        slope = 0.0
    else:
        mean = float(np.mean(counts))
        std = float(np.std(counts, ddof=0))
        slope = float(np.polyfit(years, counts, 1)[0]) if len(years) > 1 else 0.0

    province_pop = float(df['ProvincePop'].iloc[0]) if ('ProvincePop' in df.columns and not pd.isna(df['ProvincePop'].iloc[0])) else np.nan

    return pd.Series({'mean_count': mean, 'std_count': std, 'trend_slope': slope, 'province_pop': province_pop})

def load_and_prepare(crime_data, pop_data):
    crime = pd.read_csv(crime_data)
    pop = pd.read_csv(pop_data)

    year_cols = [c for c in crime.columns if '-' in c and c.split('-')[0].isdigit()]
    
    crime_long = crime.melt(id_vars=['Province','Station','Category'], value_vars=year_cols,
                            var_name='YearRange', value_name='Count')
    crime_long['YearStart'] = crime_long['YearRange'].str.split('-').str[0].astype(int)
    crime_long['Count'] = pd.to_numeric(crime_long['Count'], errors='coerce').fillna(0).astype(float)

    return crime_long, pop, sorted(year_cols, key=lambda x: int(x.split('-')[0]))

def aggregate_station_year(crime_long, pop):
    station_year = crime_long.groupby(['Province','Station','YearStart'], as_index=False)['Count'].sum()
    pop_lookup = pop.set_index('Province')['Population'].to_dict()
    station_year['ProvincePop'] = station_year['Province'].map(pop_lookup)
    return station_year

def build_station_features(station_year):
    station_features = station_year.groupby(['Province','Station']).apply(compute_station_features).reset_index()

    def rate_row(r):
        if not pd.isna(r['province_pop']) and r['province_pop'] > 0:
            return r['mean_count'] / (r['province_pop'] / 100000.0)
        return np.nan

    station_features['mean_per_100k'] = station_features.apply(rate_row, axis=1)
    return station_features

def define_hotspots(station_features, percentile=0.8):
    valid_rates = station_features['mean_per_100k'].dropna()
    threshold = valid_rates.quantile(percentile)
    station_features = station_features.copy()
    station_features['is_hotspot'] = station_features['mean_per_100k'].apply(
        lambda v: 1 if (not pd.isna(v) and v > threshold) else 0
    )
    return station_features, threshold

# Main dashboard logic
if crime_data and population_data:
    try:
        # Load and process data
        with st.spinner('Loading and processing data...'):
            crime_long, pop_df, year_cols = load_and_prepare(crime_data, population_data)
            station_year = aggregate_station_year(crime_long, pop_df)
            station_features = build_station_features(station_year)
            station_features, threshold = define_hotspots(station_features, percentile_threshold)

        # Rest of your dashboard code remains the same...
        st.header("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stations", len(station_features))
        with col2:
            st.metric("Hotspot Stations", station_features['is_hotspot'].sum())
        with col3:
            st.metric("Hotspot Threshold", f"{threshold:.2f} per 100k")

        # Continue with the rest of your analysis...
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please check your data format or try a different dataset.")
else:
    st.info("👈 Please select a data source from the sidebar to begin analysis.")
    
    # Data format guidance
    st.subheader("📋 Expected Data Format")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Crime Data Format:**")
        st.code("""
Province, Station, Category, 2005-2006, 2006-2007, ...
Western Cape, Cape Town Central, Burglary, 100, 120, ...
Gauteng, Jhb Central, Theft, 150, 130, ...
        """)
    
    with col2:
        st.markdown("**Population Data Format:**")
        st.code("""
Province, Population, Area, Density
Gauteng, 12272263, 18178, 675.1
Western Cape, 5822734, 129462, 45.0
        """)
    
    # Sample data links
    st.subheader("🌐 Sample Dataset Links")
    for name, urls in SAMPLE_DATASETS.items():
        st.markdown(f"**{name}:**")
        st.markdown(f"- Crime Data: [{urls['crime']}]({urls['crime']})")
        st.markdown(f"- Population Data: [{urls['population']}]({urls['population']})")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Crime Analysis Dashboard")
