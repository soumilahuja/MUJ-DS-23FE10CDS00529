"""Streamlit demo UI:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.pipeline import build_analyzer, load_config

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")
st.title("🔍 ReviewLens")
st.caption("LLM-powered customer review analyzer")

cfg = load_config()
cfg["provider"] = st.sidebar.selectbox("Provider", ["anthropic", "openai", "mock"],
                                       index=["anthropic", "openai", "mock"].index(cfg["provider"]))
analyzer = build_analyzer(cfg)

tab1, tab2 = st.tabs(["Single review", "Batch (CSV)"])

with tab1:
    text = st.text_area("Paste a customer review", height=120)
    if st.button("Analyze", type="primary") and text.strip():
        with st.spinner("Asking the LLM..."):
            res = analyzer.analyze_review(text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", res["sentiment"].title())
        c2.metric("Estimated rating", f"{res['rating_estimate']}/5")
        c3.metric("Urgency", res["urgency"].title())
        st.write(f"**Emotion:** {res['emotion']}  \n**Key issue:** {res['key_issue']}")
        st.table(pd.DataFrame(res["aspects"]))
        st.info(f"**Suggested reply:** {res['suggested_reply']}")

with tab2:
    up = st.file_uploader("CSV with a 'review' column", type="csv")
    if up and st.button("Analyze batch"):
        df = pd.read_csv(up)
        with st.spinner(f"Analyzing {len(df)} reviews..."):
            results = analyzer.analyze_batch(df[cfg["text_column"]].dropna().astype(str).tolist())
            stats = analyzer.aggregate(results)
            summary = analyzer.executive_summary(stats)
        st.markdown(summary)
        st.bar_chart(pd.Series(stats["sentiment_counts"]))
        st.dataframe(pd.DataFrame(results))
