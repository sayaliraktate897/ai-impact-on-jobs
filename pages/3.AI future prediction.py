import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Future Outlook", page_icon="🔮", layout="wide")

st.title("🔮 AI Future Outlook")
st.markdown("### Which industries face the most AI automation and adoption?")

@st.cache_data
def load_jobs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "archive (9)", "ai_job_market_insights.csv")
    return pd.read_csv(path)

df = load_jobs()

st.info("This dataset has no year column, so this page does not forecast. "
        "It compares how industries differ today in automation risk and AI adoption.")

choice = st.radio("Compare industries by:", ["Automation_Risk", "AI_Adoption_Level"], horizontal=True)

selected = st.multiselect("Choose industries to compare",
                          sorted(df["Industry"].unique()),
                          default=sorted(df["Industry"].unique()))

data = df[df["Industry"].isin(selected)]

if data.empty:
    st.warning("Please select at least one industry.")
    st.stop()

# -----------------------------
# Chart
# -----------------------------
st.subheader(f"{choice.replace('_', ' ')} by Industry (% of jobs)")

order = ["Low", "Medium", "High"]
colors = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}

fig = px.histogram(
    data, x="Industry", color=choice, barnorm="percent",
    category_orders={choice: order},
    color_discrete_map=colors,
)
fig.update_layout(yaxis_title="% of jobs")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Insight (calculated automatically)
# -----------------------------
high_share = (
    (data[choice] == "High").groupby(data["Industry"]).mean() * 100
).sort_values(ascending=False)

st.success(
    f"**What this shows:** {high_share.index[0]} has the highest share of "
    f"**High** {choice.replace('_', ' ').lower()} jobs ({high_share.iloc[0]:.1f}%), "
    f"and {high_share.index[-1]} has the lowest ({high_share.iloc[-1]:.1f}%)."
)

# -----------------------------
# Table
# -----------------------------
st.subheader("Table: number of jobs")
st.dataframe(pd.crosstab(data["Industry"], data[choice]), use_container_width=True)

st.caption("Source: Kaggle, AI-Powered Job Market Insights (synthetic dataset, 500 job listings).")
st.caption("Developed by: Sayali Dattatray Raktate | AI Impact in 2030")