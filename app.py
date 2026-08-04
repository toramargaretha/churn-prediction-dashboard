import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Prediksi Customer Churn E-Commerce",
    page_icon="🛒",
    layout="wide"
)


@st.cache_resource(ttl="1h")
def load_artifacts():
    model = joblib.load("model_xgboost.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, encoders, feature_names, explainer


model, scaler, encoders, feature_names, explainer = load_artifacts()


st.title("Dashboard Prediksi Customer Churn E-Commerce")
st.markdown(
    "Sistem prediksi risiko churn pelanggan berbasis model **XGBoost**, "
    "dengan penanganan data tidak seimbang melalui **SMOTE** dan interpretasi "
    "hasil prediksi menggunakan **SHAP**."
)


menu = st.sidebar.radio(
    "Navigasi",
    ["Prediksi Individual", "Prediksi Batch (CSV)", "Ringkasan Model"]
)


def segment_risk(prob):
    if prob >= 0.70:
        return "Tinggi", "🔴"
    elif prob >= 0.40:
        return "Sedang", "🟡"
    else:
        return "Rendah", "🟢"


def clean_numeric_input(df, feature_names):
    """
    Membersihkan dan menyiapkan DataFrame input agar seluruh kolom
    fitur bertipe numerik dan siap diproses oleh scaler/model.
    Mengembalikan (df_bersih, daftar_kolom_bermasalah).
    """
    df_clean = df.reindex(columns=feature_names, fill_value=0).copy()

    if "loyalty_member" in df_clean.columns:
        loyalty_map = {
            "Ya": 1, "Tidak": 0,
            "Yes": 1, "No": 0,
            "ya": 1, "tidak": 0,
            "yes": 1, "no": 0,
            1: 1, 0: 0, "1": 1, "0": 0
        }
        df_clean["loyalty_member"] = df_clean["loyalty_member"].replace(loyalty_map)

    before_numeric = df_clean.copy()
    df_clean = df_clean.apply(pd.to_numeric, errors="coerce")

    problem_cols = []
    for col in df_clean.columns:
        if df_clean[col].isna().any() and not before_numeric[col].isna().any():
            problem_cols.append(col)

    df_clean = df_clean.fillna(0)
    return df_clean, problem_cols


if menu == "Prediksi Individual":
    st.header("Input Data Pelanggan")
    col1, col2, col3 = st.columns(3)

    with col1:
        account_age = st.number_input("Account Age (bulan)", min_value=0, value=12)
        total_orders = st.number_input("Total Orders", min_value=0, value=10)
        avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=250.0)
        days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=15)
        discount_usage_rate = st.slider("Discount Usage Rate", 0.0, 1.0, 0.3)

    with col2:
        return_rate = st.slider("Return Rate", 0.0, 1.0, 0.1)
        support_tickets = st.number_input("Customer Support Tickets", min_value=0, value=1)
        loyalty_member = st.selectbox("Loyalty Member", ["Ya", "Tidak"])
        browsing_freq = st.number_input("Browsing Frequency per Week", min_value=0, value=5)
        cart_abandonment_rate = st.slider("Cart Abandonment Rate", 0.0, 1.0, 0.2)

    with col3:
        review_score = st.slider("Product Review Score Average", 0.0, 5.0, 3.5)
        engagement_score = st.slider("Engagement Score", 0.0, 1.0, 0.5)
        satisfaction_score = st.slider("Satisfaction Score", 0.0, 5.0, 3.5)
        price_sensitivity = st.slider("Price Sensitivity Index", 0.0, 1.0, 0.5)

    if st.button("Prediksi Churn", type="primary"):
        loyalty_map = {"Ya": 1, "Tidak": 0}
        loyalty_encoded = loyalty_map[loyalty_member]

        input_dict = {
            "account_age_months": account_age,
            "total_orders": total_orders,
            "avg_order_value": avg_order_value,
            "days_since_last_purchase": days_since_last_purchase,
            "discount_usage_rate": discount_usage_rate,
            "return_rate": return_rate,
            "customer_support_tickets": support_tickets,
            "loyalty_member": loyalty_encoded,
            "browsing_frequency_per_week": browsing_freq,
            "cart_abandonment_rate": cart_abandonment_rate,
            "product_review_score_avg": review_score,
            "engagement_score": engagement_score,
            "satisfaction_score": satisfaction_score,
            "price_sensitivity_index": price_sensitivity,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_df)

        proba = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]
        risk_label, risk_icon = segment_risk(proba)

        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Probabilitas Churn", f"{proba:.2%}")
            st.metric("Status Prediksi", "Churn" if pred == 1 else "Tidak Churn")
            st.metric("Segmentasi Risiko", f"{risk_icon} {risk_label}")

        with res_col2:
            st.subheader("Interpretasi SHAP")
            shap_values = explainer.shap_values(input_scaled)

            fig, ax = plt.subplots(figsize=(6, 4))
            shap.summary_plot(
                shap_values, input_scaled,
                feature_names=feature_names,
                plot_type="bar", show=False
            )
            st.pyplot(fig)

        if risk_label == "Tinggi":
            st.warning(
                "Pelanggan ini berisiko tinggi churn. Disarankan strategi "
                "win-back campaign, personalisasi notifikasi, dan penawaran "
                "loyalitas khusus."
            )
        elif risk_label == "Sedang":
            st.info(
                "Pelanggan menunjukkan indikasi risiko sedang. Pantau aktivitas "
                "transaksinya secara berkala dan pertimbangkan program "
                "keterlibatan tambahan."
            )
        else:
            st.success("Pelanggan berada pada kondisi stabil dengan risiko churn yang rendah.")

elif menu == "Prediksi Batch (CSV)":
    st.header("Prediksi Massal via Upload CSV")
    st.markdown(
    "Unggah data pelanggan dalam format CSV untuk memproses prediksi churn "
    "secara massal. Format kolom harus sesuai dengan fitur model; kolom "
    "`loyalty_member` menerima nilai 0/1 maupun teks (Ya/Tidak atau Yes/No)."
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Pratinjau Data:")
        st.dataframe(batch_df.head())

        if st.button("Jalankan Prediksi Batch"):
            missing_cols = [c for c in feature_names if c not in batch_df.columns]
            if missing_cols:
                st.warning(
                    f"Kolom berikut tidak ditemukan pada file dan akan diisi "
                    f"nilai 0 secara otomatis: {missing_cols}"
                )

            batch_input, problem_cols = clean_numeric_input(batch_df, feature_names)

            if problem_cols:
                st.error(
                    f"Kolom berikut mengandung nilai yang tidak bisa diubah "
                    f"menjadi angka dan telah diisi 0 secara otomatis: {problem_cols}. "
                    f"Mohon periksa kembali isi file CSV Anda."
                )

            try:
                batch_scaled = scaler.transform(batch_input)
            except Exception as e:
                st.error(f"Gagal memproses data untuk prediksi. Detail: {e}")
                st.stop()

            probs = model.predict_proba(batch_scaled)[:, 1]
            preds = model.predict(batch_scaled)

            batch_df["Probabilitas_Churn"] = probs
            batch_df["Prediksi"] = np.where(preds == 1, "Churn", "Tidak Churn")
            batch_df["Segmentasi_Risiko"] = batch_df["Probabilitas_Churn"].apply(
                lambda p: segment_risk(p)[0]
            )

            st.success(f"Prediksi selesai untuk {len(batch_df)} pelanggan.")
            st.dataframe(batch_df)

            csv_result = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh Hasil Prediksi (CSV)",
                data=csv_result,
                file_name="hasil_prediksi_churn.csv",
                mime="text/csv"
            )

            st.subheader("Distribusi Segmentasi Risiko")
            risk_counts = batch_df["Segmentasi_Risiko"].value_counts()
            st.bar_chart(risk_counts)


elif menu == "Ringkasan Model":
    st.header("Ringkasan Performa Model")
    st.markdown(
        "Perbandingan performa tiga model yang telah diuji pada penelitian ini: "
        "TabNet, Random Forest, dan XGBoost."
    )

    metrics_df = pd.DataFrame({
        "Model": ["TabNet", "Random Forest", "XGBoost"],
        "Accuracy": [0.9608, 0.9617, 0.9675],
        "F1-Score": [0.8753, 0.8821, 0.8976],
        "AUC-ROC": [0.9910, 0.9915, 0.9930],
    })

    st.dataframe(metrics_df, use_container_width=True)
    st.bar_chart(metrics_df.set_index("Model"))

    st.markdown(
        "Model **XGBoost** dipilih sebagai model utama dalam dashboard ini "
        "karena mencatat nilai metrik tertinggi di antara ketiga model yang diuji."
    )


st.sidebar.divider()
st.sidebar.caption(
    "Skripsi: Perbandingan Performa TabNet, Random Forest, dan XGBoost "
    "dalam Prediksi Customer Churn E-Commerce dengan SMOTE dan Interpretasi SHAP\n\n"
    "Tora Margaretha Chrisdya Wardani - 51422591"
)