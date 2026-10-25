"""
Interactive Streamlit Dashboard for Customer Review Sentiment Analysis.
Features:
- Dual Model Selector (TF-IDF ML Pipeline vs Hugging Face DistilBERT)
- Live Sentiment Analyzer with feature attribution keywords & sample reviews
- Batch CSV Analyzer with interactive charts and labeled data export
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Automatically launch Streamlit web server if executed directly via "python app.py" / IDE Run button
if __name__ == "__main__":
    from streamlit.runtime import exists
    if not exists():
        import subprocess
        sys.exit(subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]).returncode)

# Configure page metadata & layout
st.set_page_config(
    page_title="SentimentScope | Customer Review NLP Classifier",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.predict import (
    predict_sentiment,
    predict_batch,
    get_tfidf_pipeline,
    PIPELINE_PATH
)

METRICS_PATH = os.path.join(CURRENT_DIR, "models", "metrics.json")

# Custom Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e293b 0%, #3b82f6 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-card .val {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-card .lbl {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }

    .sentiment-badge-pos {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.3rem;
    }

    .sentiment-badge-neg {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #fff1f2;
        color: #9f1239;
        border: 1px solid #fecdd3;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.3rem;
    }

    .keyword-chip {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .keyword-pos {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .keyword-neg {
        background: #ffe4e6;
        color: #881337;
        border: 1px solid #fda4af;
    }

    .sidebar-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load benchmark metrics
@st.cache_data
def load_benchmark_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

metrics_data = load_benchmark_metrics()

# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Engine Configuration")
    
    # Check if local model exists
    has_local_model = os.path.exists(PIPELINE_PATH)

    model_choice = st.radio(
        "Select Active NLP Model:",
        options=[
            "Custom TF-IDF + ML Pipeline (Fast & Local)",
            "Hugging Face DistilBERT Transformer (Deep Learning)"
        ],
        index=0 if has_local_model else 1,
        help="Switch seamlessly between the lightning-fast custom TF-IDF classifier and the pre-trained DistilBERT transformer."
    )
    
    selected_model_type = "tfidf" if "TF-IDF" in model_choice else "distilbert"

    st.markdown("---")

    # Display Benchmark Metrics from train.py
    st.markdown("### 📊 Trained Model Benchmark")
    if metrics_data and "benchmark" in metrics_data:
        best_name = metrics_data.get("best_model", "Logistic Regression")
        b_metrics = metrics_data["benchmark"].get(best_name, {})
        
        st.markdown(f"**Top Model:** `{best_name}`")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", f"{b_metrics.get('accuracy', 0)*100:.1f}%")
            st.metric("Macro F1", f"{b_metrics.get('macro_f1', 0)*100:.1f}%")
        with c2:
            st.metric("ROC-AUC", f"{b_metrics.get('roc_auc', 0)*100:.1f}%")
            st.metric("Test Size", f"{metrics_data.get('test_size', 0):,}")
            
        with st.expander("🔍 View All Model Benchmarks"):
            for m_name, m_vals in metrics_data["benchmark"].items():
                st.markdown(f"**{m_name}**")
                st.write(f"- Acc: `{m_vals['accuracy']*100:.2f}%` | F1: `{m_vals['macro_f1']*100:.2f}%` | Time: `{m_vals.get('training_time_seconds', 0)}s`")
    else:
        st.info("💡 Run `python src/train.py` to benchmark models and populate real-time metrics.")

    st.markdown("---")
    st.markdown("### 🛠️ Architecture Highlights")
    st.markdown("""
    - **Preprocessing:** Negation preservation, HTML/URL stripping, WordNet lemmatization.
    - **Vectorization:** Sublinear TF-IDF (1-2 N-grams, 25k features).
    - **Transformer:** DistilBERT fine-tuned on SST-2.
    - **Status:** Production-Ready 🟢
    """)

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown('<div class="main-title">💬 SentimentScope AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Production-Grade Customer Review Sentiment Analysis & NLP Classification Engine</div>', unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2 = st.tabs([
    "🎯 Live Review Sentiment Analyzer",
    "📁 Batch CSV Dataset Analyzer"
])

# ==========================================
# TAB 1: LIVE REVIEW ANALYZER
# ==========================================
with tab1:
    st.markdown("#### Enter or paste any customer review below:")
    
    # Preset sample reviews for instant testing
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_text = ""
    
    with col_p1:
        if st.button("🌟 Preset: Glowing Positive", use_container_width=True):
            st.session_state["review_input"] = (
                "Absolutely loved this product! The build quality is top-notch, battery life is phenomenal, "
                "and it exceeded all my expectations. 10/10 recommend to anyone!"
            )
    with col_p2:
        if st.button("💔 Preset: Critical Negative", use_container_width=True):
            st.session_state["review_input"] = (
                "Terrible experience. The package arrived damaged, customer service was extremely unhelpful, "
                "and the device stopped working after just three days. Total waste of money!"
            )
    with col_p3:
        if st.button("🎭 Preset: Tricky / Mixed Review", use_container_width=True):
            st.session_state["review_input"] = (
                "I was initially hesitant because setup seemed tricky, but once configured it is actually not bad at all. "
                "Works surprisingly well for the price point!"
            )

    default_text = st.session_state.get(
        "review_input",
        "The customer service was remarkably fast and resolved my issue immediately. Highly impressed with the quality!"
    )

    user_text = st.text_area(
        "Customer Review Text:",
        value=default_text,
        height=140,
        placeholder="Type or paste your customer feedback, movie review, or product comment here..."
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)
        
    with col_info:
        st.caption(f"Active Engine: **{model_choice}**")

    if analyze_btn or user_text:
        if not user_text.strip():
            st.warning("Please type a review to analyze.")
        else:
            try:
                # Fallback check if user picks TF-IDF but model hasn't been trained yet
                active_type = selected_model_type
                if active_type == "tfidf" and not os.path.exists(PIPELINE_PATH):
                    st.warning("⚠️ Local TF-IDF model not yet trained. Automatically falling back to DistilBERT Transformer.")
                    active_type = "distilbert"

                with st.spinner("Analyzing review semantics and tone..."):
                    result = predict_sentiment(user_text, model_type=active_type)

                sentiment = result["sentiment"]
                confidence = result["confidence"]
                pos_p = result["probabilities"]["Positive"]
                neg_p = result["probabilities"]["Negative"]

                st.markdown("---")
                
                # Results layout
                res_col1, res_col2 = st.columns([1.2, 2])

                with res_col1:
                    st.markdown("##### Prediction Result")
                    if sentiment == "Positive":
                        st.markdown(
                            f'<div class="sentiment-badge-pos">😊 Positive Sentiment</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="sentiment-badge-neg">😠 Negative Sentiment</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence Level:** `{confidence * 100:.2f}%`")
                    st.progress(float(confidence))

                    # Quick review metrics
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Word Count", result["word_count"])
                    with m2:
                        st.metric("Char Count", len(user_text))

                with res_col2:
                    st.markdown("##### Probability Breakdown")
                    chart_df = pd.DataFrame({
                        "Sentiment": ["Positive 😊", "Negative 😠"],
                        "Probability": [pos_p, neg_p]
                    })
                    
                    bar_chart = alt.Chart(chart_df).mark_bar(cornerRadius=6).encode(
                        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                        y=alt.Y("Sentiment:N", sort=None, title=""),
                        color=alt.Color("Sentiment:N", scale=alt.Scale(
                            domain=["Positive 😊", "Negative 😠"],
                            range=["#10b981", "#ef4444"]
                        ), legend=None),
                        tooltip=["Sentiment", alt.Tooltip("Probability:Q", format=".2%")]
                    ).properties(height=140)
                    
                    st.altair_chart(bar_chart, use_container_width=True)

                # Feature attribution (Keyword Explainability)
                if result.get("keywords"):
                    st.markdown("##### 🔍 Top Influential Keywords Driving This Decision")
                    st.caption("Extracted via TF-IDF token weights and model feature coefficients:")
                    
                    kw_html = ""
                    for kw in result["keywords"]:
                        cls_name = "keyword-pos" if kw["sentiment"] == "Positive" else "keyword-neg"
                        icon = "▲" if kw["sentiment"] == "Positive" else "▼"
                        kw_html += f'<span class="keyword-chip {cls_name}">{icon} {kw["word"]} ({kw["score"]:+.2f})</span>'
                    
                    st.markdown(kw_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during inference: {e}")

# ==========================================
# TAB 2: BATCH CSV ANALYSIS
# ==========================================
with tab2:
    st.markdown("#### Analyze a Batch of Customer Reviews via CSV")
    st.markdown("Upload any CSV dataset containing customer reviews, product feedback, or comments.")

    # Demo CSV download option
    with st.expander("💡 Don't have a CSV handy? Click to generate and download a sample batch test file"):
        sample_df = pd.DataFrame({
            "review_id": [101, 102, 103, 104, 105, 106],
            "customer_review": [
                "Outstanding quality! Exceeded expectations in every single way.",
                "Terrible customer service and the device broke on day two.",
                "Super fast shipping, works seamlessly right out of the package.",
                "Not worth the price. Low quality plastic and very noisy.",
                "Brilliant design, intuitive software, five stars all the way!",
                "Complete waste of time and money, extremely frustrating experience."
            ]
        })
        st.dataframe(sample_df, use_container_width=True)
        csv_sample = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sample Reviews CSV",
            data=csv_sample,
            file_name="sample_reviews.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV file:",
        type=["csv"],
        help="Upload a CSV file containing at least one column with review text."
    )

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded `{len(df_batch):,}` rows from `{uploaded_file.name}`")
            
            # Select review column
            possible_cols = [c for c in df_batch.columns if df_batch[c].dtype == object]
            default_col = possible_cols[0] if possible_cols else df_batch.columns[0]
            
            for candidate in ["review", "customer_review", "text", "comment", "feedback"]:
                if candidate in df_batch.columns:
                    default_col = candidate
                    break

            c_col, c_limit = st.columns([2, 1])
            with c_col:
                selected_col = st.selectbox("Select Text Review Column:", options=df_batch.columns, index=list(df_batch.columns).index(default_col))
            with c_limit:
                max_rows = st.number_input("Max Rows to Process (0 = All):", min_value=0, max_value=len(df_batch), value=min(1000, len(df_batch)))

            if max_rows > 0:
                df_to_process = df_batch.head(max_rows).copy()
            else:
                df_to_process = df_batch.copy()

            if st.button("🚀 Run Batch Sentiment Classification", type="primary"):
                # Fallback check
                active_type = selected_model_type
                if active_type == "tfidf" and not os.path.exists(PIPELINE_PATH):
                    st.warning("⚠️ Local TF-IDF model not yet trained. Using DistilBERT Transformer.")
                    active_type = "distilbert"

                with st.spinner(f"Classifying {len(df_to_process):,} reviews with {active_type.upper()}..."):
                    df_result = predict_batch(df_to_process, text_column=selected_col, model_type=active_type)

                st.markdown("---")
                st.markdown("### 📈 Batch Analysis Summary")

                # Metrics KPIs
                total_cnt = len(df_result)
                pos_cnt = (df_result["predicted_sentiment"] == "Positive").sum()
                neg_cnt = (df_result["predicted_sentiment"] == "Negative").sum()
                pos_pct = (pos_cnt / total_cnt) * 100 if total_cnt > 0 else 0
                neg_pct = (neg_cnt / total_cnt) * 100 if total_cnt > 0 else 0
                avg_conf = df_result["confidence"].mean() * 100 if "confidence" in df_result.columns else 0

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.markdown(f'<div class="metric-card"><div class="val">{total_cnt:,}</div><div class="lbl">Total Reviews</div></div>', unsafe_allow_html=True)
                with kpi2:
                    st.markdown(f'<div class="metric-card"><div class="val" style="color: #10b981;">{pos_pct:.1f}%</div><div class="lbl">Positive ({pos_cnt:,})</div></div>', unsafe_allow_html=True)
                with kpi3:
                    st.markdown(f'<div class="metric-card"><div class="val" style="color: #ef4444;">{neg_pct:.1f}%</div><div class="lbl">Negative ({neg_cnt:,})</div></div>', unsafe_allow_html=True)
                with kpi4:
                    st.markdown(f'<div class="metric-card"><div class="val" style="color: #3b82f6;">{avg_conf:.1f}%</div><div class="lbl">Avg Confidence</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Charts
                ch_col1, ch_col2 = st.columns(2)
                
                with ch_col1:
                    st.markdown("##### Sentiment Ratio (Donut Chart)")
                    pie_data = pd.DataFrame({
                        "Sentiment": ["Positive", "Negative"],
                        "Count": [pos_cnt, neg_cnt]
                    })
                    donut = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Sentiment", type="nominal", scale=alt.Scale(
                            domain=["Positive", "Negative"],
                            range=["#10b981", "#ef4444"]
                        )),
                        tooltip=["Sentiment", "Count"]
                    ).properties(height=260)
                    st.altair_chart(donut, use_container_width=True)

                with ch_col2:
                    st.markdown("##### Sentiment Count Comparison")
                    bar = alt.Chart(pie_data).mark_bar(cornerRadius=6).encode(
                        x=alt.X("Sentiment:N", title=""),
                        y=alt.Y("Count:Q", title="Number of Reviews"),
                        color=alt.Color("Sentiment:N", scale=alt.Scale(
                            domain=["Positive", "Negative"],
                            range=["#10b981", "#ef4444"]
                        ), legend=None),
                        tooltip=["Sentiment", "Count"]
                    ).properties(height=260)
                    st.altair_chart(bar, use_container_width=True)

                # Filterable Data Table
                st.markdown("##### 📋 Labeled Review Results")
                filter_choice = st.selectbox("Filter Display by Sentiment:", ["All", "Positive Only", "Negative Only"])
                
                if filter_choice == "Positive Only":
                    display_df = df_result[df_result["predicted_sentiment"] == "Positive"]
                elif filter_choice == "Negative Only":
                    display_df = df_result[df_result["predicted_sentiment"] == "Negative"]
                else:
                    display_df = df_result

                st.dataframe(display_df, use_container_width=True, height=350)

                # Download Labeled CSV
                csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Labeled CSV",
                    data=csv_bytes,
                    file_name="sentiment_classified_results.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
