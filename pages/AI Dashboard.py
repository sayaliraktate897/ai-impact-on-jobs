import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Dashboard", page_icon="📊", layout="wide")

st.title("🤖 AI Impact on Jobs Dashboard")
st.markdown("### Job market overview across industries")

@st.cache_data
def load_jobs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "archive (9)", "ai_job_market_insights.csv")
    return pd.read_csv(path)

df = load_jobs()

# -----------------------------
# Filter
# -----------------------------
choice = st.selectbox("🔍 Select Industry", ["All"] + sorted(df["Industry"].unique()))
data = df if choice == "All" else df[df["Industry"] == choice]

# -----------------------------
# KPI cards (calculated from the selection)
# -----------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("💼 Job listings", len(data))
k2.metric("💰 Avg salary", f"${data['Salary_USD'].mean():,.0f}")
k3.metric("📈 Growing jobs", f"{(data['Job_Growth_Projection'] == 'Growth').mean() * 100:.1f}%")
k4.metric("⚠️ High automation risk", f"{(data['Automation_Risk'] == 'High').mean() * 100:.1f}%")
k5.metric("🤖 Common AI adoption", data["AI_Adoption_Level"].mode().iloc[0])
k6.metric("🏠 Remote-friendly", f"{(data['Remote_Friendly'] == 'Yes').mean() * 100:.1f}%")

st.write("---")

# -----------------------------
# Row 1: outlook (filtered) + job count by industry
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader(f"📊 Job Outlook: {choice}")
    outlook = data["Job_Growth_Projection"].value_counts().reset_index()
    outlook.columns = ["Outlook", "Jobs"]
    fig1 = px.pie(outlook, names="Outlook", values="Jobs", hole=0.5,
                  color="Outlook",
                  color_discrete_map={"Growth": "#2ecc71", "Stable": "#f1c40f", "Decline": "#e74c3c"})
    st.plotly_chart(fig1, width="stretch")

with right:
    st.subheader("🥧 Job Listings by Industry")
    counts = df["Industry"].value_counts().reset_index()
    counts.columns = ["Industry", "Jobs"]
    fig2 = px.pie(counts, names="Industry", values="Jobs", hole=0.5)
    st.plotly_chart(fig2, width="stretch")

st.write("---")

# -----------------------------
# Row 2: growth outlook by industry (replaces the made-up 2025-2030 chart)
# -----------------------------
st.subheader("📈 Growth Outlook by Industry (% of jobs)")
fig3 = px.histogram(df, x="Industry", color="Job_Growth_Projection", barnorm="percent",
                    category_orders={"Job_Growth_Projection": ["Growth", "Stable", "Decline"]},
                    color_discrete_map={"Growth": "#2ecc71", "Stable": "#f1c40f", "Decline": "#e74c3c"})
fig3.update_layout(yaxis_title="% of jobs", legend_title="Outlook")
st.plotly_chart(fig3, width="stretch")

st.write("---")

# -----------------------------
# Row 3: top industries by salary
# -----------------------------
st.subheader("🏆 Top Industries by Average Salary (USD)")
sal = (df.groupby("Industry")["Salary_USD"].mean().round(0)
         .sort_values().reset_index())
fig4 = px.bar(sal, x="Salary_USD", y="Industry", orientation="h",
              color="Salary_USD", color_continuous_scale="Viridis")
st.plotly_chart(fig4, width="stretch")

st.caption("Source: Kaggle, AI-Powered Job Market Insights (synthetic dataset, 500 job listings).")
st.caption("Developed by: Sayali Dattatray Raktate | AI Impact in 2030")