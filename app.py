"""
PUBG Random Forest Regressor - Streamlit Web Application
=========================================================
Interactive web app untuk analisis dan prediksi PUBG Win Place Percentage
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PUBG Random Forest Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set theme to light
st.markdown("""
    <style>
    /* Light theme styling */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Main content area */
    .main {
        background-color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #262730;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    h2 {
        color: #ff6b00;
        border-bottom: 2px solid #ff6b00;
        padding-bottom: 10px;
    }
    h3 {
        color: #0066cc;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 25px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background-color: #ffffff;
    }
    
    /* Text inputs */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #262730;
        border: 1px solid #e0e0e0;
    }
    
    /* Number inputs */
    .stNumberInput>div>div>input {
        background-color: #ffffff;
        color: #262730;
        border: 1px solid #e0e0e0;
    }
    
    /* Sliders */
    .stSlider>div>div>div {
        background-color: #f0f2f6;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #262730;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #f0f2f6;
        color: #262730;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🎮 PUBG Random Forest Predictor")
st.markdown("### Prediksi Win Place Percentage dengan Machine Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 5em;'>🎮</h1>", unsafe_allow_html=True)
    st.title("⚙️ Configuration")
    
    # Data sampling
    st.subheader("📊 Data Sampling")
    sample_size = st.slider(
        "Jumlah Samples",
        min_value=10000,
        max_value=200000,
        value=50000,
        step=10000,
        help="Jumlah data yang digunakan untuk training"
    )
    
    # Model parameters
    st.subheader("🤖 Model Parameters")
    n_estimators = st.slider("Number of Trees", 50, 200, 100, 10)
    max_depth = st.slider("Max Depth", 5, 30, 15, 1)
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
    
    st.markdown("---")
    st.info("💡 **Tip**: Gunakan sample size lebih kecil untuk training lebih cepat")

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('PUBG_Game_Prediction_data.csv')
    return df

@st.cache_data
def preprocess_data(df, sample_size):
    """Preprocess and feature engineering"""
    # Drop missing target
    df = df.dropna(subset=['winPlacePerc'])
    
    # Drop unnecessary columns
    df = df.drop(['Id', 'groupId', 'matchId'], axis=1)
    
    # Encode matchType
    le = LabelEncoder()
    df['matchType_encoded'] = le.fit_transform(df['matchType'])
    df = df.drop('matchType', axis=1)
    
    # Feature engineering
    df['totalDistance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    df['healsAndBoosts'] = df['heals'] + df['boosts']
    df['killsNorm'] = df['kills'] * ((100 - df['numGroups']) / 100 + 1)
    df['damageDealtNorm'] = df['damageDealt'] * ((100 - df['numGroups']) / 100 + 1)
    
    # Sample data
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    return df_sample

@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Main app
try:
    # Load data
    with st.spinner("🔄 Loading data..."):
        df_raw = load_data()
    
    st.success(f"✅ Dataset loaded: {len(df_raw):,} rows")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Performance", 
        "🎯 Predictions",
        "📚 Documentation"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df_raw):,}")
        with col2:
            st.metric("Total Columns", df_raw.shape[1])
        with col3:
            st.metric("Missing Values", df_raw.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.markdown("---")
        
        # Dataset preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df_raw.head(100), use_container_width=True, height=300)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Descriptive Statistics")
            st.dataframe(df_raw.describe(), use_container_width=True)
        
        with col2:
            st.subheader("🎯 Target Distribution")
            fig = px.histogram(
                df_raw.sample(10000), 
                x='winPlacePerc',
                nbins=50,
                title="Distribution of Win Place Percentage",
                color_discrete_sequence=['#ff4b4b']
            )
            fig.update_layout(
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(color='#262730')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Model Training
    with tab2:
        st.header("🤖 Model Training")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("🔄 Preprocessing data..."):
                df_processed = preprocess_data(df_raw, sample_size)
                
                # Prepare data
                X = df_processed.drop('winPlacePerc', axis=1)
                y = df_processed['winPlacePerc']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Store in session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = X.columns.tolist()
            
            st.success(f"✅ Data prepared: {len(X_train):,} training samples, {len(X_test):,} test samples")
            
            with st.spinner("🔄 Training Random Forest model..."):
                model = train_model(X_train, y_train, n_estimators, max_depth)
                st.session_state['model'] = model
            
            st.success("✅ Model training completed!")
            
            # Training info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", f"{len(X_train):,}")
            with col2:
                st.metric("Test Samples", f"{len(X_test):,}")
            with col3:
                st.metric("Features", X.shape[1])
            
            # Feature importance
            st.subheader("🎯 Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='Importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Click the button above to train the model")
    
    # TAB 3: Performance
    with tab3:
        st.header("📈 Model Performance")
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Display metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test R² Score", f"{test_r2:.4f}", 
                         delta=f"{(test_r2 - train_r2):.4f}")
            with col2:
                st.metric("Test RMSE", f"{test_rmse:.4f}",
                         delta=f"{(test_rmse - train_rmse):.4f}", delta_color="inverse")
            with col3:
                st.metric("Test MAE", f"{test_mae:.4f}",
                         delta=f"{(test_mae - train_mae):.4f}", delta_color="inverse")
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted
                sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test.iloc[sample_idx],
                    y=y_test_pred[sample_idx],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=y_test.iloc[sample_idx],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Actual")
                    ),
                    name='Predictions'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='Perfect Prediction'
                ))
                fig.update_layout(
                    title=f"Actual vs Predicted (R² = {test_r2:.4f})",
                    xaxis_title="Actual winPlacePerc",
                    yaxis_title="Predicted winPlacePerc",
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    font=dict(color='#262730'),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residuals
                residuals = y_test.iloc[sample_idx] - y_test_pred[sample_idx]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test_pred[sample_idx],
                    y=residuals,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=residuals,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Residual")
                    ),
                    name='Residuals'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Residual Plot",
                    xaxis_title="Predicted winPlacePerc",
                    yaxis_title="Residuals",
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    font=dict(color='#262730'),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution comparison
            st.subheader("📊 Distribution Comparison")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=y_test,
                name='Actual',
                opacity=0.7,
                marker_color='blue',
                nbinsx=50
            ))
            fig.add_trace(go.Histogram(
                x=y_test_pred,
                name='Predicted',
                opacity=0.7,
                marker_color='red',
                nbinsx=50
            ))
            fig.update_layout(
                title="Distribution: Actual vs Predicted",
                xaxis_title="winPlacePerc",
                yaxis_title="Frequency",
                barmode='overlay',
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(color='#262730'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab")
    
    # TAB 4: Predictions
    with tab4:
        st.header("🎯 Make Predictions")
        
        if 'model' in st.session_state:
            st.subheader("🎮 Enter Player Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                kills = st.number_input("Kills", 0, 50, 2)
                assists = st.number_input("Assists", 0, 20, 0)
                damageDealt = st.number_input("Damage Dealt", 0.0, 5000.0, 150.0)
                headshotKills = st.number_input("Headshot Kills", 0, 30, 0)
                DBNOs = st.number_input("DBNOs", 0, 20, 0)
            
            with col2:
                walkDistance = st.number_input("Walk Distance", 0.0, 10000.0, 1500.0)
                rideDistance = st.number_input("Ride Distance", 0.0, 20000.0, 0.0)
                swimDistance = st.number_input("Swim Distance", 0.0, 5000.0, 0.0)
                heals = st.number_input("Heals", 0, 50, 2)
                boosts = st.number_input("Boosts", 0, 30, 3)
            
            with col3:
                weaponsAcquired = st.number_input("Weapons Acquired", 0, 50, 3)
                killPlace = st.number_input("Kill Place", 1, 100, 50)
                longestKill = st.number_input("Longest Kill", 0.0, 1000.0, 0.0)
                revives = st.number_input("Revives", 0, 20, 0)
                roadKills = st.number_input("Road Kills", 0, 10, 0)
            
            if st.button("🔮 Predict Win Place", type="primary"):
                # Create input dataframe
                input_data = {
                    'assists': assists,
                    'boosts': boosts,
                    'damageDealt': damageDealt,
                    'DBNOs': DBNOs,
                    'headshotKills': headshotKills,
                    'heals': heals,
                    'killPlace': killPlace,
                    'killPoints': 0,
                    'kills': kills,
                    'killStreaks': 0,
                    'longestKill': longestKill,
                    'matchDuration': 1800,
                    'maxPlace': 100,
                    'numGroups': 50,
                    'rankPoints': 1500,
                    'revives': revives,
                    'rideDistance': rideDistance,
                    'roadKills': roadKills,
                    'swimDistance': swimDistance,
                    'teamKills': 0,
                    'vehicleDestroys': 0,
                    'walkDistance': walkDistance,
                    'weaponsAcquired': weaponsAcquired,
                    'winPoints': 0,
                    'matchType_encoded': 0,
                    'totalDistance': walkDistance + rideDistance + swimDistance,
                    'healsAndBoosts': heals + boosts,
                    'killsNorm': kills,
                    'damageDealtNorm': damageDealt
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Ensure correct column order
                input_df = input_df[st.session_state['feature_names']]
                
                # Predict
                prediction = st.session_state['model'].predict(input_df)[0]
                
                # Display result
                st.markdown("---")
                st.subheader("🎊 Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Win Place %", f"{prediction*100:.2f}%")
                with col2:
                    rank = int((1 - prediction) * 100)
                    st.metric("Estimated Rank", f"#{rank}/100")
                with col3:
                    if prediction > 0.8:
                        performance = "🏆 Excellent!"
                    elif prediction > 0.5:
                        performance = "👍 Good"
                    else:
                        performance = "💪 Keep Trying"
                    st.metric("Performance", performance)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Win Place Percentage"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(255,255,255,1)',
                    font=dict(color='#262730'),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab")
    
    # TAB 5: Documentation
    with tab5:
        st.header("📚 Documentation")
        
        st.markdown("""
        ## 🎮 PUBG Random Forest Regressor
        
        ### 📖 Tentang Aplikasi
        Aplikasi ini menggunakan **Random Forest Regressor** untuk memprediksi **Win Place Percentage** 
        dalam game PUBG berdasarkan statistik permainan pemain.
        
        ### ✅ Keunggulan Random Forest
        - ✅ **Menangani banyak fitur** dengan baik (29+ features)
        - ✅ **Robust terhadap outlier** - tidak sensitif terhadap nilai ekstrem
        - ✅ **Mudah diimplementasikan** - tidak perlu feature scaling
        - ✅ **Baseline kuat** - R² score ~90% adalah performa excellent
        - ✅ **Feature importance** - memberikan insight fitur paling berpengaruh
        
        ### 📊 Fitur Utama
        1. **Data Overview**: Eksplorasi dataset dan statistik
        2. **Model Training**: Training model dengan parameter yang dapat disesuaikan
        3. **Performance**: Evaluasi performa model dengan berbagai metrik
        4. **Predictions**: Prediksi real-time berdasarkan input pengguna
        
        ### 🎯 Top Features yang Berpengaruh
        - **walkDistance**: Jarak berjalan (23.5%)
        - **killPlace**: Ranking kills (22.2%)
        - **totalDistance**: Total mobilitas (19.2%)
        - **boosts**: Penggunaan boost items (7.3%)
        - **weaponsAcquired**: Jumlah senjata (6.4%)
        
        ### 📈 Performance Metrics
        - **R² Score**: ~0.90 (90% variance explained)
        - **RMSE**: ~0.095 (error rata-rata 9.5%)
        - **MAE**: ~0.068 (absolute error 6.8%)
        
        ### 💡 Tips Penggunaan
        1. Mulai dengan sample size kecil (50k) untuk training cepat
        2. Tingkatkan jumlah trees untuk akurasi lebih baik
        3. Gunakan tab Predictions untuk testing model
        4. Perhatikan feature importance untuk strategi game
        
        ### 🚀 Next Steps
        Untuk meningkatkan akurasi:
        - Gunakan algoritma boosting (XGBoost, LightGBM)
        - Hyperparameter tuning lebih detail
        - Feature engineering lebih lanjut
        - Ensemble dengan model lain
        
        ---
        
        **Developed with ❤️ using Streamlit & scikit-learn**
        """)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("💡 Pastikan file 'PUBG_Game_Prediction_data.csv' ada di folder yang sama")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>🎮 PUBG Random Forest Predictor | Machine Learning Course 2025</p>
        <p>Built with Streamlit 🚀</p>
    </div>
""", unsafe_allow_html=True)
