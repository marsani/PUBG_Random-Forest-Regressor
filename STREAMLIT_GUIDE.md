# 🎮 PUBG Random Forest Streamlit App - Quick Start

## ✅ Perbaikan yang Telah Dilakukan

### 1. **Background Putih → Dark Theme** ✓
- Menambahkan konfigurasi tema dark di `.streamlit/config.toml`
- Memperbarui CSS untuk memaksa dark mode di semua komponen
- Background sekarang: `#0e1117` (dark)
- Sidebar: `#262730` (dark gray)

### 2. **Gambar yang Tidak Jalan** ✓
- Menghapus link gambar eksternal yang broken
- Menggantinya dengan emoji icon 🎮 yang lebih reliable
- Tidak memerlukan koneksi internet

## 🚀 Cara Menjalankan

```bash
# Masuk ke folder project
cd "/Users/mac/Documents/Materi Pengajaran-2025/04. Machine Learning/latihan/PUBG_Random Forest Regressor"

# Jalankan aplikasi
python3 -m streamlit run app.py
```

Aplikasi akan terbuka di: **http://localhost:8501**

## 📱 Fitur Aplikasi

### Tab 1: 📊 Data Overview
- Statistik dataset lengkap
- Preview data
- Distribusi target variable

### Tab 2: 🤖 Model Training
- Training Random Forest dengan parameter custom
- Feature importance visualization
- Real-time training progress

### Tab 3: 📈 Performance
- Metrics: R², RMSE, MAE
- Actual vs Predicted plot
- Residual analysis
- Distribution comparison

### Tab 4: 🎯 Predictions
- Input statistik pemain manual
- Prediksi win place percentage
- Gauge visualization
- Performance rating

### Tab 5: 📚 Documentation
- Panduan lengkap penggunaan
- Penjelasan algoritma
- Tips & tricks

## ⚙️ Konfigurasi di Sidebar

- **Sample Size**: 10k - 200k (default: 50k)
- **Number of Trees**: 50 - 200 (default: 100)
- **Max Depth**: 5 - 30 (default: 15)
- **Test Size**: 10% - 40% (default: 20%)

## 🎨 Tema Dark Mode

File konfigurasi: `.streamlit/config.toml`

```toml
[theme]
primaryColor="#ff4b4b"        # Merah untuk buttons & highlights
backgroundColor="#0e1117"      # Background utama (dark)
secondaryBackgroundColor="#262730"  # Sidebar & cards (dark gray)
textColor="#fafafa"           # Text putih
```

## 💡 Tips Penggunaan

1. **Training Cepat**: Gunakan sample size 50k untuk eksperimen
2. **Akurasi Maksimal**: Tingkatkan ke 100k+ samples
3. **Lihat Feature Importance**: Untuk strategi game yang lebih baik
4. **Coba Predictions**: Test dengan statistik berbeda

## 🔧 Troubleshooting

### Jika background masih putih:
1. Refresh browser (Ctrl/Cmd + R)
2. Clear cache browser
3. Restart aplikasi Streamlit

### Jika ada error:
1. Pastikan semua dependencies terinstall: `pip3 install -r requirements.txt`
2. Pastikan file CSV ada di folder yang sama
3. Check terminal untuk error messages

## 📊 Performance

- **Loading Time**: ~2-3 detik
- **Training Time**: ~2-5 detik (50k samples)
- **Prediction Time**: Instant (<1 detik)

---

**Status**: ✅ Aplikasi berjalan dengan dark theme!  
**URL**: http://localhost:8501
