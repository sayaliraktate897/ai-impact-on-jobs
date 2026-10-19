import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Industries", page_icon="🏭", layout="wide")

st.title("🏭 AI Impact Across Industries")
st.markdown("### What does a job-market dataset tell us about AI and industries?")

# -----------------------------
# Load data (one dataset for the whole page)
# -----------------------------
@st.cache_data
def load_jobs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "archive (9)", "ai_job_market_insights.csv")
    return pd.read_csv(path)

df = load_jobs()

# -----------------------------
# About the data
# -----------------------------
with st.expander("ℹ️ About the data (read this first)", expanded=True):
    st.write(f"""
**Source:** Kaggle, *AI-Powered Job Market Insights* (synthetic dataset)

**Size:** {len(df)} job listings across {df['Industry'].nunique()} industries

**Key columns used:**
- **Industry**: the sector of the job
- **Job_Growth_Projection**: Growth, Stable or Decline
- **Salary_USD**: salary of the job
- **Remote_Friendly**: whether the job can be done remotely

**Note:** The dataset is synthetic (generated for learning), so the results show
how I analyse data, not real-world statistics.
""")

# -----------------------------
# Calculations
# -----------------------------
summary = df.groupby("Industry").agg(
    Jobs=("Industry", "count"),
    Avg_Salary=("Salary_USD", "mean"),
    Growth_Share=("Job_Growth_Projection", lambda s: (s == "Growth").mean() * 100),
    Remote_Share=("Remote_Friendly", lambda s: (s == "Yes").mean() * 100),
).round(1).reset_index()

best_growth = summary.sort_values("Growth_Share", ascending=False).iloc[0]
worst_growth = summary.sort_values("Growth_Share").iloc[0]
best_salary = summary.sort_values("Avg_Salary", ascending=False).iloc[0]

# -----------------------------
# Key insights
# -----------------------------
st.subheader("🔑 Key Insights")
c1, c2, c3 = st.columns(3)
c1.metric("Highest growth outlook", best_growth["Industry"], f"{best_growth['Growth_Share']}% growing jobs", delta_color="off")
c2.metric("Lowest growth outlook", worst_growth["Industry"], f"{worst_growth['Growth_Share']}% growing jobs", delta_color="off")
c3.metric("Highest average salary", best_salary["Industry"], f"${best_salary['Avg_Salary']:,.0f}", delta_color="off")

st.write("---")

# -----------------------------
# Chart 1: growth
# -----------------------------
st.subheader("📈 Share of Jobs with a Growth Outlook (%)")
st.caption("For each industry: out of all job listings, how many are projected to Grow.")

growth = summary.sort_values("Growth_Share", ascending=False)
fig1 = px.bar(growth, x="Industry", y="Growth_Share",
              color="Growth_Share", color_continuous_scale="Teal",
              labels={"Growth_Share": "Growing jobs (%)"})
st.plotly_chart(fig1, width="stretch")
st.info(f"**What this shows:** {best_growth['Industry']} leads with {best_growth['Growth_Share']}% "
        f"of jobs growing, while {worst_growth['Industry']} is lowest at {worst_growth['Growth_Share']}%.")

st.write("---")

# -----------------------------
# Chart 2: salary
# -----------------------------
st.subheader("💰 Average Salary by Industry (USD)")
salary = summary.sort_values("Avg_Salary", ascending=False)
fig2 = px.bar(salary, x="Industry", y="Avg_Salary",
              color="Avg_Salary", color_continuous_scale="Blues",
              labels={"Avg_Salary": "Average salary (USD)"})
st.plotly_chart(fig2, width="stretch")
st.info(f"**What this shows:** {best_salary['Industry']} pays the most on average "
        f"(${best_salary['Avg_Salary']:,.0f}).")

st.write("---")

# -----------------------------
# Full table
# -----------------------------
st.subheader("📋 Industry Summary Table")
st.dataframe(summary.sort_values("Growth_Share", ascending=False), width="stretch")

st.caption("Source: Kaggle, AI-Powered Job Market Insights (synthetic dataset).")
st.caption("Developed by: Sayali Dattatray Raktate")