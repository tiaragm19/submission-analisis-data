# Dashboard Analisis E-Commerce

Dashboard Streamlit untuk submission Proyek Analisis Data (E-Commerce Public Dataset).
Menjawab tiga pertanyaan bisnis: kategori produk dengan revenue tertinggi, keterlambatan pengiriman per state beserta dampaknya ke review score, dan segmentasi pelanggan dengan RFM.

## Struktur folder
- `notebook.ipynb` : proses analisis data lengkap
- `dashboard/dashboard.py` : kode dashboard
- `dashboard/main_data.csv` : data hasil cleaning (output notebook)
- `data/` : dataset mentah (CSV)
- `requirements.txt` : daftar library

## Menjalankan dashboard di lokal
```
pip install -r requirements.txt
streamlit run dashboard/dashboard.py
```
