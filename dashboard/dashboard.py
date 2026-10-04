import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "main_data.csv")
KOLOM = [
    "order_id", "price", "product_category_name_english", "order_purchase_timestamp",
    "customer_unique_id", "customer_state", "is_late", "delay_days", "review_score",
]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, usecols=KOLOM, parse_dates=["order_purchase_timestamp"])
    df["order_month"] = df.order_purchase_timestamp.dt.to_period("M")
    return df


def hitung_rfm(df):
    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("price", "sum"),
        )
        .reset_index()
    )
    ref_date = df.order_purchase_timestamp.max().normalize() + pd.Timedelta(days=1)
    rfm["recency"] = (ref_date - rfm.last_purchase).dt.days

    try:
        rfm["r_score"] = pd.qcut(rfm.recency, 3, labels=[3, 2, 1]).astype(int)
        rfm["m_score"] = pd.qcut(rfm.monetary, 3, labels=[1, 2, 3]).astype(int)
    except ValueError:  # batas kuantil kembar pada data yang sangat kecil
        rfm["r_score"] = pd.qcut(rfm.recency.rank(method="first"), 3, labels=[3, 2, 1]).astype(int)
        rfm["m_score"] = pd.qcut(rfm.monetary.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)

    def segmen(row):
        if row.frequency >= 2:
            return "Loyal Customers"
        if row.r_score == 3:
            return "New High Spenders" if row.m_score == 3 else "Recent Customers"
        return "At-Risk High Spenders" if row.m_score == 3 else "Lost / Low Value"

    rfm["segment"] = rfm.apply(segmen, axis=1)
    ringkasan = rfm.groupby("segment").agg(
        customers=("customer_unique_id", "count"),
        revenue=("monetary", "sum"),
        avg_recency=("recency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    ringkasan["customer_pct"] = ringkasan.customers / ringkasan.customers.sum() * 100
    ringkasan["revenue_pct"] = ringkasan.revenue / ringkasan.revenue.sum() * 100
    return ringkasan.sort_values("revenue_pct", ascending=False).reset_index(), ref_date


# ---------------------------------------------------------------- data & filter
if not os.path.exists(DATA_PATH):
    st.error("File main_data.csv tidak ditemukan. Letakkan di folder yang sama dengan dashboard.py.")
    st.stop()

main_df = load_data()

st.sidebar.header("Filter")
tgl_min = main_df.order_purchase_timestamp.min().date()
tgl_max = main_df.order_purchase_timestamp.max().date()
rentang = st.sidebar.date_input("Rentang tanggal pesanan", value=(tgl_min, tgl_max),
                                min_value=tgl_min, max_value=tgl_max)
semua_state = sorted(main_df.customer_state.unique())
state_pilihan = st.sidebar.multiselect("State pelanggan", semua_state, default=semua_state)

if not isinstance(rentang, (tuple, list)) or len(rentang) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir pada filter di samping.")
    st.stop()

mulai, akhir = pd.Timestamp(rentang[0]), pd.Timestamp(rentang[1]) + pd.Timedelta(days=1)
df = main_df[
    (main_df.order_purchase_timestamp >= mulai)
    & (main_df.order_purchase_timestamp < akhir)
    & (main_df.customer_state.isin(state_pilihan))
]
if df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

order_df = df.drop_duplicates(subset="order_id")

# ---------------------------------------------------------------- header & metrik
st.title("🛒 Dashboard Analisis E-Commerce")
st.caption("Pesanan berstatus delivered, Januari 2017 - Agustus 2018. Revenue = total price (tanpa ongkir).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total revenue", f"{df.price.sum():,.0f}")
k2.metric("Jumlah pesanan", f"{order_df.order_id.nunique():,}")
k3.metric("Pesanan terlambat", f"{order_df.is_late.mean() * 100:.2f}%")
k4.metric("Rata-rata review score", f"{order_df.review_score.mean():.2f}")

tab1, tab2, tab3 = st.tabs(["Kategori Produk", "Keterlambatan Pengiriman", "Segmentasi Pelanggan (RFM)"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Kategori produk apa yang menyumbang revenue terbesar?")
    kategori = (
        df.groupby("product_category_name_english")
        .agg(revenue=("price", "sum"), total_orders=("order_id", "nunique"), avg_price=("price", "mean"))
        .sort_values("revenue", ascending=False)
    )
    kategori["share_pct"] = kategori.revenue / kategori.revenue.sum() * 100

    n_top = st.slider("Jumlah kategori teratas yang ditampilkan", 3, 15, 10)
    top_n = kategori.head(n_top).reset_index()
    warna = ["#1f77b4" if i < 5 else "#c7c7c7" for i in range(len(top_n))]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_n, x="revenue", y="product_category_name_english",
                palette=warna, hue="product_category_name_english", legend=False, ax=ax)
    ax.set_title(f"{n_top} Kategori dengan Revenue Tertinggi (5 teratas disorot)", fontsize=13, loc="left")
    ax.set_xlabel("Total revenue (price)")
    ax.set_ylabel("")
    ax.ticklabel_format(style="plain", axis="x")
    for i, (rev, pct) in enumerate(zip(top_n.revenue, top_n.share_pct)):
        ax.text(rev, i, f"  {rev / 1e6:.2f} jt ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, top_n.revenue.max() * 1.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Tren revenue bulanan 5 kategori teratas")
    top5 = kategori.head(5).index
    bulanan = (
        df[df.product_category_name_english.isin(top5)]
        .groupby(["order_month", "product_category_name_english"])
        .price.sum().unstack().fillna(0)
    )
    bulanan.index = bulanan.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in bulanan.columns:
        ax.plot(bulanan.index, bulanan[col], marker="o", markersize=3, linewidth=2, label=col)
    ax.set_title("Tren Revenue Bulanan 5 Kategori Teratas", fontsize=13, loc="left")
    ax.set_xlabel("Bulan pesanan")
    ax.set_ylabel("Revenue (price)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Kategori", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat tabel ringkasan kategori"):
        st.dataframe(kategori.head(n_top).round(2))

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Di state mana pesanan paling sering terlambat?")
    state_df = (
        order_df.groupby("customer_state")
        .agg(total_orders=("order_id", "nunique"), late_pct=("is_late", "mean"))
        .sort_values("late_pct", ascending=False)
    )
    state_df["late_pct"] = state_df.late_pct * 100

    min_pesanan = st.slider("Minimal jumlah pesanan per state", 0, 2000, 500, step=50,
                            help="State dengan pesanan sedikit membuat persentase kurang stabil.")
    state_plot = state_df[state_df.total_orders >= min_pesanan].head(10).reset_index()

    if state_plot.empty:
        st.info("Tidak ada state yang memenuhi batas minimal pesanan pada filter ini.")
    else:
        nasional = order_df.is_late.mean() * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=state_plot, x="late_pct", y="customer_state", color="#d62728", ax=ax)
        ax.axvline(nasional, color="black", linestyle="--", linewidth=1,
                   label=f"Rata-rata pada filter {nasional:.2f}%")
        ax.legend(loc="lower right")
        for i, (pct, n) in enumerate(zip(state_plot.late_pct, state_plot.total_orders)):
            ax.text(pct, i, f"  {pct:.1f}% (n={n:,})", va="center", fontsize=9)
        ax.set_title(f"State dengan Persentase Pesanan Terlambat Tertinggi (minimal {min_pesanan} pesanan)",
                     fontsize=13, loc="left")
        ax.set_xlabel("Pesanan tiba melewati estimasi (%)")
        ax.set_ylabel("State pelanggan")
        ax.set_xlim(0, state_plot.late_pct.max() * 1.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Seberapa besar keterlambatan memengaruhi review score?")
    grup = pd.cut(
        order_df.delay_days, bins=[-np.inf, 0, 3, 7, np.inf],
        labels=["Tepat waktu /\nlebih awal", "Terlambat\n1-3 hari", "Terlambat\n4-7 hari", "Terlambat\n>7 hari"],
    )
    review_grup = (
        order_df.assign(delay_group=grup)
        .groupby("delay_group", observed=True)
        .agg(avg_review=("review_score", "mean"), total_orders=("order_id", "count"))
        .reset_index()
    )
    review_grup["delay_group"] = review_grup.delay_group.astype(str)
    warna_r = {
        "Tepat waktu /\nlebih awal": "#2ca02c", "Terlambat\n1-3 hari": "#ff9896",
        "Terlambat\n4-7 hari": "#d62728", "Terlambat\n>7 hari": "#8c1d1d",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=review_grup, x="delay_group", y="avg_review", palette=warna_r,
                hue="delay_group", dodge=False, legend=False, ax=ax)
    for i, v in enumerate(review_grup.avg_review):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_title("Semakin Lama Terlambat, Semakin Rendah Review Score", fontsize=13, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Rata-rata review score (skala 1-5)")
    ax.set_ylim(0, 5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Grafik menunjukkan hubungan, bukan bukti sebab-akibat.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Segmen pelanggan mana yang paling bernilai?")
    segmen_df, ref_date = hitung_rfm(df)
    st.caption(
        f"Pelanggan diidentifikasi dengan customer_unique_id. Tanggal acuan recency: {ref_date.date()}. "
        "Recency dan Monetary dibagi 3 kelompok kuantil, Frequency dipisah menjadi 1 kali vs 2 kali atau lebih. "
        "Segmen dihitung ulang sesuai filter."
    )

    panjang = segmen_df.melt(id_vars="segment", value_vars=["customer_pct", "revenue_pct"],
                             var_name="ukuran", value_name="persen")
    panjang["ukuran"] = panjang.ukuran.map({"customer_pct": "% pelanggan", "revenue_pct": "% revenue"})
    teratas = segmen_df.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=panjang, x="persen", y="segment", hue="ukuran",
                palette={"% pelanggan": "#c7c7c7", "% revenue": "#1f77b4"}, ax=ax)
    for p in ax.patches:
        w = p.get_width()
        if w > 0:
            ax.text(w + 0.5, p.get_y() + p.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)
    ax.set_title(f"{teratas.segment}: {teratas.customer_pct:.0f}% pelanggan, {teratas.revenue_pct:.0f}% revenue",
                 fontsize=13, loc="left")
    ax.set_xlabel("Persentase dari total (pelanggan atau revenue)")
    ax.set_ylabel("")
    ax.set_xlim(0, panjang.persen.max() * 1.15)
    ax.legend(title="", loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat tabel ringkasan segmen"):
        st.dataframe(segmen_df.round(2))

st.caption("Sumber data: E-Commerce Public Dataset. Dashboard ini dibuat untuk submission analisis data.")
