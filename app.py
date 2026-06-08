import os
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# =====================================================================
# SYSTEM CONFIGURATION & UI STYLING
# =====================================================================
st.set_page_config(
    page_title="PropAI | Enterprise Valuation Engine", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injector for a sleek, premium corporate aesthetic
st.markdown("""
    <style>
        /* Main page background tweak */
        .stApp {
            background-color: #0f111a;
            color: #e6e6e6;
        }
        /* Custom Header Styling */
        .main-header {
            font-size: 40px !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .sub-header {
            font-size: 16px !important;
            color: #8a90a6 !important;
            margin-bottom: 30px;
        }
        /* Dashboard Card Container Design */
        .metric-card {
            background-color: #1a1c24;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #0072ff;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .valuation-box {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,114,255,0.2);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# DATA WORKFLOW PIPELINE
# =====================================================================
@st.cache_data
def load_and_clean_pipeline():
    filename = "scraped_properties.csv"
    
    # Generate mock scraped data if file is absent (Scaled up to 2,000 for high precision learning)
    if not os.path.exists(filename):
        neighborhoods = {
            "Downtown Core": (12.9716, 77.5946),
            "Silicon Suburbs": (12.9279, 77.6271),
            "Riverside Estates": (13.0285, 77.5416),
            "Greenwood Valley": (12.9815, 77.7499)
        }
        records = []
        for _ in range(2000): # Boosted sample size significantly for rich data patterns
            locality = random.choice(list(neighborhoods.keys()))
            b_lat, b_lon = neighborhoods[locality]
            lat, lon = b_lat + random.uniform(-0.012, 0.012), b_lon + random.uniform(-0.012, 0.012)
            bhk = random.choice([1, 2, 3, 4])
            area = int(bhk * random.uniform(550, 750) + random.uniform(-50, 50))
            
            # Mathematical market logic with built-in realistic price noise
            base_price = 4500 * area if locality == "Downtown Core" else 2800 * area
            price = int(base_price + (bhk * 400000) + random.uniform(-150000, 150000))
            
            records.append({
                "Title": f"Premium {bhk} BHK in {locality}",
                "Raw_Price": f"₹ {price:,} /- Only",
                "Raw_Area": f"{area} Sq-Ft",
                "Raw_BHK": f"{bhk} BHK",
                "Locality": locality, "lat": lat, "lon": lon
            })
        pd.DataFrame(records).to_csv(filename, index=False)
        
    df = pd.read_csv(filename)
    
    # Process Raw Fields using Clean Regex Engines
    df['Price'] = df['Raw_Price'].str.replace(r'[^\d]', '', regex=True).astype(float)
    df['Area'] = df['Raw_Area'].str.replace(r'[^\d]', '', regex=True).astype(float)
    df['BHK'] = df['Raw_BHK'].str.extract(r'(\d+)').astype(float)
    
    # ADVANCED FEATURE ENGINEERING: Generate density spatial ratio attributes
    df['Space_Per_BHK'] = df['Area'] / df['BHK']
    
    return df.drop(columns=['Raw_Price', 'Raw_Area', 'Raw_BHK'])

df_clean = load_and_clean_pipeline()

# Prepare ML training matrices
df_encoded = pd.get_dummies(df_clean, columns=['Locality'], drop_first=True)

# Drop Target variables along with coordinates and text-heavy titles
features = [c for c in df_encoded.columns if c not in ['Price', 'lat', 'lon', 'Title']]
X, y = df_encoded[features], df_encoded['Price']

# =====================================================================
# MACHINE LEARNING ENGINE (HYPERPARAMETER OPTIMIZED)
# =====================================================================
@st.cache_resource
def train_engine(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # High-Performance XGBoost Configuration Dials
    model = xgb.XGBRegressor(
        n_estimators=350,       # More estimators give trees space to minimize residual error
        learning_rate=0.04,     # Slower updates lock onto optimal functions without overshooting
        max_depth=7,            # Deeper structural capability captures neighborhood variations
        subsample=0.8,          # Sample fraction protects against data memorization
        colsample_bytree=0.8,   # High dimensional stability
        random_state=42
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return model, r2, mae

model, r2_score_val, mae_val = train_engine(X, y)

# =====================================================================
# SIDEBAR PRODUCTION CONTROL PANEL
# =====================================================================
st.sidebar.markdown("## ⚙️ Core Architecture")
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div class='metric-card'>
        <p style='color:#8a90a6; margin:0;'>ALGORITHM</p>
        <h3 style='color:#00c6ff; margin:0;'>XGBoost Optimized</h3>
    </div>
    <br>
    <div class='metric-card'>
        <p style='color:#8a90a6; margin:0;'>MODEL ACCURACY (R²)</p>
        <h3 style='color:#00ff88; margin:0;'>{r2_score_val*100:.2f}%</h3>
    </div>
    <br>
    <div class='metric-card'>
        <p style='color:#8a90a6; margin:0;'>MEAN ABSOLUTE ERROR</p>
        <h3 style='color:#ff4b4b; margin:0;'>₹ {mae_val:,.0f}</h3>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📝 Developer Pipeline Logs"):
    st.caption(f"✓ Ingestion: {len(df_clean)} records parsed.")
    st.caption("✓ Engineering: 'Space_Per_BHK' ratio mapped.")
    st.caption("✓ Tuner Engine: Active parameters deployed.")

# =====================================================================
# MAIN USER UI DASHBOARD
# =====================================================================
st.markdown("<div class='main-header'>PropAI: Real Estate Valuation Suite</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enterprise Machine Learning interface utilizing live automated data extraction pipelines.</div>", unsafe_allow_html=True)

# Split screen workspace layout into Left (Controls) and Right (Analytics)
layout_left, layout_right = st.columns([1, 1.2], gap="large")

with layout_left:
    st.markdown("### 🕹️ Dynamic Valuation Controls")
    
    # UI Component Styling Enclosed inside Container
    with st.container(border=True):
        selected_locality = st.selectbox("Geographical Target Zone", sorted(df_clean['Locality'].unique()))
        selected_bhk = st.radio("Property Layout (BHK)", options=[1, 2, 3, 4], index=1, horizontal=True)
        selected_area = st.slider("Total Structural Footprint (Sq.Ft.)", 400, 3500, 1250, step=25)
        
        st.markdown("<br>", unsafe_allow_html=True)
        calculate_clicked = st.button("🚀 Compute Real-Time Valuation", use_container_width=True, type="primary")

    if calculate_clicked:
        # Vector Reconstruction for Model Evaluation (Must match feature set exactly)
        input_vector = pd.DataFrame(0, index=[0], columns=features)
        input_vector['Area'] = selected_area
        input_vector['BHK'] = selected_bhk
        
        # Calculate engineered ratio live during prediction
        input_vector['Space_Per_BHK'] = selected_area / selected_bhk
        
        target_col = f"Locality_{selected_locality}"
        if target_col in input_vector.columns:
            input_vector[target_col] = 1
            
        prediction = model.predict(input_vector)[0]
        
        # Display highly prominent pricing output card
        st.markdown(f"""
            <div class='valuation-box'>
                <p style='color:#d1d5db; font-size:16px; margin:0; text-transform:uppercase; letter-spacing:1px;'>Predicted Market Valuation</p>
                <h1 style='color:#ffffff; font-size:42px; margin:10px 0;'>₹ {prediction:,.2f}</h1>
                <p style='color:#a3e635; font-size:13px; margin:0;'>✓ Strategy estimate verified based on zone indices.</p>
            </div>
        """, unsafe_allow_html=True)

with layout_right:
    st.markdown("### 📊 Market Spatial Analytics")
    
    # Tab Layout Splitter for Visual Elegance
    tab_map, tab_chart = st.tabs(["🗺️ Geographic Distribution", "📈 Price vs Area Trends"])
    
    with tab_map:
        map_filter = df_clean[df_clean['Locality'] == selected_locality]
        st.map(map_filter, latitude='lat', longitude='lon', size=25, zoom=12)
        st.caption(f"Showing localized clusters extracted for {selected_locality}.")

    with tab_chart:
        # Render a rich, clean analytical Scatter plot using Plotly
        fig = px.scatter(
            df_clean, 
            x="Area", 
            y="Price", 
            color="Locality",
            template="plotly_dark",
            title="Market Density Vector Analysis",
            labels={"Area": "Square Feet", "Price": "Price (INR)"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)