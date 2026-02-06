# PUBG Game Prediction - Random Forest Regressor

## 📊 Overview

Implementasi **Random Forest Regressor** untuk memprediksi **Win Place Percentage** dalam game PUBG (PlayerUnknown's Battlegrounds). Model ini berfungsi sebagai **baseline yang kuat** untuk perbandingan dengan algoritma boosting.

## Rancangan Aplikasi 
<img width="1388" height="784" alt="image" src="https://github.com/user-attachments/assets/b6d77912-dfe8-429b-8429-5ca5f0c8553a" />

<img width="1369" height="783" alt="image" src="https://github.com/user-attachments/assets/32517f98-a5ca-428e-a391-1404c2028109" />

## 🎯 Tujuan

Memprediksi peringkat kemenangan pemain (winPlacePerc) berdasarkan statistik permainan seperti:
- Jarak yang ditempuh (walkDistance, rideDistance, swimDistance)
- Statistik pertempuran (kills, damage, headshots)
- Item yang digunakan (heals, boosts, weapons)
- Informasi match (matchType, numGroups, maxPlace)

## 📁 Dataset

- **File**: `PUBG_Game_Prediction_data.csv` Link : https://www.kaggle.com/datasets/ashishjangra27/pubg-games-dataset
- **Total Records**: 4,446,966 rows
- **Features**: 29 kolom
- **Target**: `winPlacePerc` (0.0 - 1.0)
- **Samples Used**: 100,000 (untuk efisiensi training)

## 🔧 Preprocessing & Feature Engineering

### Data Cleaning
- Menghapus missing values pada target variable
- Menghapus kolom ID yang tidak relevan (Id, groupId, matchId)
- Encoding categorical variable (matchType) menggunakan LabelEncoder

### Feature Engineering
Membuat fitur baru untuk meningkatkan performa model:
- `totalDistance`: Total jarak yang ditempuh (walk + ride + swim)
- `healsAndBoosts`: Total item healing yang digunakan
- `killsNorm`: Kills yang dinormalisasi berdasarkan jumlah grup
- `damageDealtNorm`: Damage yang dinormalisasi berdasarkan jumlah grup

## 🤖 Model Configuration

```python
RandomForestRegressor(
    n_estimators=100,        # 100 decision trees
    max_depth=15,            # Kedalaman maksimal tree
    min_samples_split=10,    # Minimum samples untuk split
    min_samples_leaf=5,      # Minimum samples di leaf node
    max_features='sqrt',     # √n features untuk setiap split
    random_state=42,
    n_jobs=-1               # Parallel processing
)
```

## 📈 Performance Metrics

### Training Set
- **RMSE**: 0.0800
- **MAE**: 0.0572
- **R² Score**: 0.9322

### Test Set
- **RMSE**: 0.0949
- **MAE**: 0.0683
- **R² Score**: 0.9033

### Cross-Validation (5-fold)
- **Mean R²**: 0.9009 (±0.0040)

## 🎯 Top 5 Most Important Features

| Feature | Importance |
|---------|-----------|
| walkDistance | 0.2349 |
| killPlace | 0.2225 |
| totalDistance | 0.1918 |
| boosts | 0.0726 |
| weaponsAcquired | 0.0644 |

## ✅ Keunggulan Random Forest

1. **Menangani Banyak Fitur**: Dapat bekerja dengan 29+ fitur tanpa masalah
2. **Robust terhadap Outlier**: Tidak sensitif terhadap nilai ekstrem
3. **Implementasi Mudah**: Tidak memerlukan feature scaling atau normalisasi
4. **Baseline Kuat**: R² score 90.33% adalah baseline yang sangat baik
5. **Feature Importance**: Memberikan insight tentang fitur yang paling berpengaruh

## ⚠️ Keterbatasan

1. **Akurasi**: Meskipun bagus, masih kalah dibanding algoritma boosting (XGBoost, LightGBM)
2. **Model Size**: Ensemble dari 100 trees membutuhkan memory yang cukup besar
3. **Inference Speed**: Lebih lambat dibanding model tunggal untuk prediksi

## 🚀 Cara Menjalankan

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script
python3 pubg_random_forest.py
```

## 📊 Output

Script akan menghasilkan:
1. **Console Output**: Detailed analysis dan metrics
2. **Visualization**: `pubg_random_forest_analysis.png` dengan 6 plots:
   - Feature Importance
   - Actual vs Predicted
   - Residual Plot
   - Distribution Comparison
   - Performance Metrics
   - Error Distribution

## 🔄 Next Steps

Untuk meningkatkan performa lebih lanjut:

1. **Hyperparameter Tuning**: GridSearchCV atau RandomizedSearchCV
2. **More Feature Engineering**: Interaksi antar fitur, polynomial features
3. **Ensemble Methods**: Stacking dengan model lain
4. **Boosting Algorithms**: XGBoost, LightGBM, CatBoost untuk akurasi lebih tinggi

## 📝 Kesimpulan

Random Forest Regressor memberikan **baseline yang sangat kuat** dengan R² score **90.33%**. Model ini:
- ✅ Mudah diimplementasikan
- ✅ Robust dan reliable
- ✅ Memberikan interpretability melalui feature importance
- ✅ Cocok sebagai benchmark untuk model yang lebih kompleks

Untuk production atau kompetisi, pertimbangkan algoritma boosting untuk akurasi maksimal.

---

**Author**: Machine Learning Course 2025  
**Date**: February 2026  
**Algorithm**: Random Forest Regressor  
**Framework**: scikit-learn
