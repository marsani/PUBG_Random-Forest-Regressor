"""
PUBG Game Prediction - Random Forest Regressor
================================================
Dataset: PUBG Game Prediction Data
Algorithm: Random Forest Regressor
Target: winPlacePerc (Win Place Percentage)

Keunggulan Random Forest:
- Bisa menangani banyak fitur
- Tidak sensitif terhadap outlier
- Mudah diimplementasikan
- Baseline yang kuat untuk perbandingan dengan boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("PUBG GAME PREDICTION - RANDOM FOREST REGRESSOR")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv('PUBG_Game_Prediction_data.csv')
print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[2] EXPLORATORY DATA ANALYSIS")
print("-" * 80)

# Info dataset
print("\n2.1 Dataset Info:")
print(df.info())

# Statistik deskriptif
print("\n2.2 Descriptive Statistics:")
print(df.describe())

# Missing values
print("\n2.3 Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# Target variable distribution
print("\n2.4 Target Variable (winPlacePerc) Distribution:")
print(df['winPlacePerc'].describe())

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n[3] DATA PREPROCESSING")
print("-" * 80)

# Handle missing values in target
print(f"\n3.1 Handling missing values in target variable...")
initial_rows = len(df)
df = df.dropna(subset=['winPlacePerc'])
print(f"✓ Rows removed: {initial_rows - len(df):,}")
print(f"✓ Remaining rows: {len(df):,}")

# Drop unnecessary columns
print("\n3.2 Dropping unnecessary columns...")
columns_to_drop = ['Id', 'groupId', 'matchId']
df = df.drop(columns=columns_to_drop)
print(f"✓ Dropped columns: {columns_to_drop}")

# Encode categorical variable (matchType)
print("\n3.3 Encoding categorical variable (matchType)...")
le = LabelEncoder()
df['matchType_encoded'] = le.fit_transform(df['matchType'])
df = df.drop('matchType', axis=1)
print(f"✓ matchType encoded with {len(le.classes_)} unique values")

# Handle remaining missing values
print("\n3.4 Handling remaining missing values...")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"✓ Filled {col} with median")

# Feature Engineering - Create new features
print("\n3.5 Feature Engineering...")
df['totalDistance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
df['healsAndBoosts'] = df['heals'] + df['boosts']
df['killsNorm'] = df['kills'] * ((100 - df['numGroups']) / 100 + 1)
df['damageDealtNorm'] = df['damageDealt'] * ((100 - df['numGroups']) / 100 + 1)
print("✓ Created features: totalDistance, healsAndBoosts, killsNorm, damageDealtNorm")

# Sampling untuk efisiensi (gunakan 10% data untuk training lebih cepat)
print("\n3.6 Sampling data for efficiency...")
sample_size = min(100000, len(df))  # Maksimal 100k rows
df_sample = df.sample(n=sample_size, random_state=42)
print(f"✓ Using {sample_size:,} samples ({(sample_size/len(df)*100):.2f}% of total data)")

# ============================================================================
# 4. PREPARE DATA FOR MODELING
# ============================================================================
print("\n[4] PREPARING DATA FOR MODELING")
print("-" * 80)

# Separate features and target
X = df_sample.drop('winPlacePerc', axis=1)
y = df_sample['winPlacePerc']

print(f"\n4.1 Features shape: {X.shape}")
print(f"4.2 Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n4.3 Train-Test Split:")
print(f"   - Training set: {X_train.shape[0]:,} samples")
print(f"   - Test set: {X_test.shape[0]:,} samples")

# ============================================================================
# 5. RANDOM FOREST REGRESSOR MODEL
# ============================================================================
print("\n[5] TRAINING RANDOM FOREST REGRESSOR")
print("-" * 80)

# Initialize Random Forest Regressor
print("\n5.1 Model Configuration:")
rf_model = RandomForestRegressor(
    n_estimators=100,        # Jumlah trees
    max_depth=15,            # Kedalaman maksimal tree
    min_samples_split=10,    # Minimum samples untuk split
    min_samples_leaf=5,      # Minimum samples di leaf
    max_features='sqrt',     # Jumlah features untuk split
    random_state=42,
    n_jobs=-1,               # Gunakan semua CPU cores
    verbose=1
)

print(f"   - n_estimators: {rf_model.n_estimators}")
print(f"   - max_depth: {rf_model.max_depth}")
print(f"   - min_samples_split: {rf_model.min_samples_split}")
print(f"   - min_samples_leaf: {rf_model.min_samples_leaf}")
print(f"   - max_features: {rf_model.max_features}")

# Train model
print("\n5.2 Training model...")
rf_model.fit(X_train, y_train)
print("✓ Model training completed!")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n[6] MODEL EVALUATION")
print("-" * 80)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n6.1 Performance Metrics:")
print(f"\n   Training Set:")
print(f"   - RMSE: {train_rmse:.4f}")
print(f"   - MAE:  {train_mae:.4f}")
print(f"   - R²:   {train_r2:.4f}")

print(f"\n   Test Set:")
print(f"   - RMSE: {test_rmse:.4f}")
print(f"   - MAE:  {test_mae:.4f}")
print(f"   - R²:   {test_r2:.4f}")

# Cross-validation
print("\n6.2 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(
    rf_model, X_train, y_train, 
    cv=5, 
    scoring='r2',
    n_jobs=-1
)
print(f"   - CV R² Scores: {cv_scores}")
print(f"   - Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n[7] FEATURE IMPORTANCE ANALYSIS")
print("-" * 80)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n[8] GENERATING VISUALIZATIONS...")
print("-" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Feature Importance Plot
ax1 = plt.subplot(2, 3, 1)
top_features = feature_importance.head(15)
sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis', ax=ax1)
ax1.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
ax1.set_xlabel('Importance', fontsize=12)
ax1.set_ylabel('Feature', fontsize=12)

# 2. Actual vs Predicted (Test Set)
ax2 = plt.subplot(2, 3, 2)
sample_indices = np.random.choice(len(y_test), size=min(1000, len(y_test)), replace=False)
ax2.scatter(y_test.iloc[sample_indices], y_test_pred[sample_indices], 
            alpha=0.5, s=20, c='blue', edgecolors='k', linewidth=0.5)
ax2.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual winPlacePerc', fontsize=12)
ax2.set_ylabel('Predicted winPlacePerc', fontsize=12)
ax2.set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}', 
              fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Residuals Plot
ax3 = plt.subplot(2, 3, 3)
residuals = y_test.iloc[sample_indices] - y_test_pred[sample_indices]
ax3.scatter(y_test_pred[sample_indices], residuals, 
            alpha=0.5, s=20, c='green', edgecolors='k', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted winPlacePerc', fontsize=12)
ax3.set_ylabel('Residuals', fontsize=12)
ax3.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Distribution of Predictions
ax4 = plt.subplot(2, 3, 4)
ax4.hist(y_test, bins=50, alpha=0.5, label='Actual', color='blue', edgecolor='black')
ax4.hist(y_test_pred, bins=50, alpha=0.5, label='Predicted', color='red', edgecolor='black')
ax4.set_xlabel('winPlacePerc', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Model Performance Comparison
ax5 = plt.subplot(2, 3, 5)
metrics = ['RMSE', 'MAE', 'R²']
train_values = [train_rmse, train_mae, train_r2]
test_values = [test_rmse, test_mae, test_r2]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax5.bar(x_pos - width/2, train_values, width, label='Train', 
                color='skyblue', edgecolor='black')
bars2 = ax5.bar(x_pos + width/2, test_values, width, label='Test', 
                color='lightcoral', edgecolor='black')

ax5.set_xlabel('Metrics', fontsize=12)
ax5.set_ylabel('Score', fontsize=12)
ax5.set_title('Model Performance: Train vs Test', fontsize=14, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(metrics)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 6. Error Distribution
ax6 = plt.subplot(2, 3, 6)
errors = np.abs(y_test - y_test_pred)
ax6.hist(errors, bins=50, color='orange', edgecolor='black', alpha=0.7)
ax6.axvline(errors.mean(), color='red', linestyle='--', lw=2, 
            label=f'Mean Error: {errors.mean():.4f}')
ax6.set_xlabel('Absolute Error', fontsize=12)
ax6.set_ylabel('Frequency', fontsize=12)
ax6.set_title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pubg_random_forest_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: pubg_random_forest_analysis.png")

# ============================================================================
# 9. SUMMARY & CONCLUSIONS
# ============================================================================
print("\n[9] SUMMARY & CONCLUSIONS")
print("=" * 80)

print(f"""
RANDOM FOREST REGRESSOR - BASELINE MODEL

Dataset Summary:
- Total samples used: {sample_size:,}
- Features: {X.shape[1]}
- Target: winPlacePerc (Win Place Percentage)

Model Performance:
- Test R² Score: {test_r2:.4f}
- Test RMSE: {test_rmse:.4f}
- Test MAE: {test_mae:.4f}

Top 5 Most Important Features:
{feature_importance.head(5).to_string(index=False)}

Keunggulan Random Forest:
✓ Menangani {X.shape[1]} fitur dengan baik
✓ Robust terhadap outlier
✓ Implementasi mudah dan cepat
✓ Baseline yang kuat untuk perbandingan

Catatan:
- Model ini adalah baseline yang kuat
- Untuk akurasi lebih tinggi, pertimbangkan algoritma boosting
  (XGBoost, LightGBM, CatBoost)
- Feature engineering lebih lanjut dapat meningkatkan performa
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED!")
print("=" * 80)
