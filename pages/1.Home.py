import json
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="AI Impact on Jobs", layout="wide")

ROOT = Path(__file__).parent.parent

# ------------------------
# Load data
# ------------------------
@st.cache_data
def load_jobs():
    return pd.read_csv(ROOT / "data" / "archive (9)" / "ai_job_market_insights.csv")

df = load_jobs()

growth = ((df["Job_Growth_Projection"] == "Growth").groupby(df["Industry"]).mean() * 100).sort_values(ascending=False)
high_risk = ((df["Automation_Risk"] == "High").groupby(df["Industry"]).mean() * 100).sort_values(ascending=False)
avg_salary = df.groupby("Industry")["Salary_USD"].mean()

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1200px;}
.card{background:#1C2333;border-radius:15px;padding:25px;text-align:center;color:white;
      box-shadow:0px 4px 12px rgba(0,0,0,.4);}
.metric{font-size:40px;font-weight:bold;color:#00C8FF;}
.metric_title{font-size:18px;}
.about{background:#1C2333;padding:25px;border-radius:15px;color:white;line-height:1.8;}
img{border-radius:20px;}
</style>
""", unsafe_allow_html=True)

def card(value, title):
    return f"""
    <div class="card">
        <div class="metric">{value}</div>
        <div class="metric_title">{title}</div>
    </div>
    """

# ------------------------
# Interactive banner (numbers come from the dataset)
# ------------------------
bars_json = json.dumps([{"n": k, "v": round(float(v), 1)} for k, v in growth.items()])
total_growth = round((df["Job_Growth_Projection"] == "Growth").mean() * 100, 1)

BANNER_HTML = """
<style>
  #wrap{position:relative;width:100%;min-height:380px;border-radius:20px;overflow:hidden;
        background:linear-gradient(120deg,#0a1123,#12285a);font-family:Arial,Helvetica,sans-serif;
        display:flex;flex-wrap:wrap;align-items:center;justify-content:space-around;
        padding:24px;box-sizing:border-box;gap:20px;}
  .left{flex:1 1 320px;color:#fff;}
  .title{color:#00C8FF;font-size:50px;font-weight:800;text-shadow:0 0 22px rgba(0,200,255,.6);}
  .sub{font-size:20px;margin:8px 0 22px;color:#dbe7f5;}
  .stats{display:flex;gap:14px;flex-wrap:wrap;}
  .st{background:rgba(255,255,255,.07);border:1px solid rgba(0,200,255,.35);
      border-radius:14px;padding:12px 18px;min-width:90px;}
  .num{color:#00C8FF;font-size:30px;font-weight:700;}
  .cap{font-size:13px;color:#9fb3c8;}
  .right{flex:1 1 380px;}
  .chart{display:flex;align-items:flex-end;gap:8px;}
  .col{flex:1;display:flex;flex-direction:column;align-items:center;}
  .box{height:190px;width:100%;display:flex;align-items:flex-end;}
  .bar{width:100%;height:0;border-radius:6px 6px 0 0;
       background:linear-gradient(180deg,#00E0FF,#0068D6);transition:height 1s ease,filter .2s;}
  .lab{writing-mode:vertical-rl;transform:rotate(180deg);font-size:12px;color:#cfe0f2;
       margin-top:6px;height:135px;}
  #info{color:#00C8FF;font-size:15px;min-height:22px;margin-bottom:8px;}
  .hint{color:#9fb3c8;font-size:12px;margin-top:6px;}
</style>
<div id="wrap">
  <div class="left">
    <div class="title">AI Impact on Jobs</div>
    <div class="sub">Exploring AI, jobs and automation risk across industries</div>
    <div class="stats">
      <div class="st"><div class="num" data-t="__N__" data-s="">0</div><div class="cap">Job listings</div></div>
      <div class="st"><div class="num" data-t="__I__" data-s="">0</div><div class="cap">Industries</div></div>
      <div class="st"><div class="num" data-t="__G__" data-s="%">0</div><div class="cap">Jobs with growth outlook</div></div>
    </div>
  </div>
  <div class="right">
    <div id="info">Growth outlook by industry (hover over a bar)</div>
    <div class="chart" id="bars"></div>
    <div class="hint">Source: Kaggle dataset (synthetic, 500 job listings)</div>
  </div>
</div>
<script>
const bars = __BARS__;
const wrap = document.getElementById("bars");
const info = document.getElementById("info");
const start = info.textContent;
const max = Math.max(...bars.map(b => b.v));

bars.forEach((b, i) => {
  const col = document.createElement("div"); col.className = "col";
  const box = document.createElement("div"); box.className = "box";
  const bar = document.createElement("div"); bar.className = "bar";
  const lab = document.createElement("div"); lab.className = "lab"; lab.textContent = b.n;
  box.appendChild(bar); col.appendChild(box); col.appendChild(lab); wrap.appendChild(col);
  col.addEventListener("mouseenter", () => {
    info.textContent = b.n + ": " + b.v + "% of jobs have a growth outlook";
    bar.style.filter = "brightness(1.4)";
  });
  col.addEventListener("mouseleave", () => { info.textContent = start; bar.style.filter = ""; });
  setTimeout(() => { bar.style.height = (b.v / max * 100) + "%"; }, 300 + i * 120);
});

document.querySelectorAll(".num").forEach(el => {
  const t = parseFloat(el.dataset.t), s = el.dataset.s, t0 = performance.now();
  function step(now) {
    const p = Math.min((now - t0) / 1500, 1);
    const v = t * p;
    el.textContent = (t % 1 === 0 ? Math.round(v) : v.toFixed(1)) + s;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
});
</script>
"""

BANNER_HTML = (BANNER_HTML.replace("__BARS__", bars_json)
               .replace("__N__", str(len(df)))
               .replace("__I__", str(df["Industry"].nunique()))
               .replace("__G__", str(total_growth)))

components.html(BANNER_HTML, height=460)

st.write("")

# ------------------------
# KPI cards (from the dataset)
# ------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(card(len(df), "Job Listings Analysed"), unsafe_allow_html=True)
c2.markdown(card(df["Industry"].nunique(), "Industries"), unsafe_allow_html=True)
c3.markdown(card(f"{(df['Job_Growth_Projection'] == 'Growth').mean() * 100:.1f}%", "Jobs with Growth Outlook"),
            unsafe_allow_html=True)
c4.markdown(card(f"${df['Salary_USD'].mean():,.0f}", "Average Salary (USD)"), unsafe_allow_html=True)

st.write("")

# ------------------------
# Key findings (calculated)
# ------------------------
st.markdown("## 🔑 Key Findings")
f1, f2, f3 = st.columns(3)
f1.success(f"**Growth:** {growth.index[0]} has the highest share of growing jobs ({growth.iloc[0]:.1f}%), "
           f"and {growth.index[-1]} the lowest ({growth.iloc[-1]:.1f}%).")
f2.warning(f"**Automation risk:** {high_risk.index[0]} has the highest share of high-risk jobs "
           f"({high_risk.iloc[0]:.1f}%).")
f3.info(f"**Salary:** {avg_salary.idxmax()} pays the most on average (${avg_salary.max():,.0f}).")

st.write("")

# ------------------------
# Industries at a glance
# ------------------------
st.markdown("## 🏭 Industries at a glance")

shown = [
    ("Healthcare", "healthcare"), ("Education", "education"), ("Finance", "banking"),
    ("Manufacturing", "manufacturing"), ("Retail", "retail"), ("Transportation", "transportation"),
]

for i, (label, file) in enumerate(shown):
    if i % 3 == 0:
        cols = st.columns(3)
    img_path = ROOT / "images" / f"{file}.jpeg"
    with cols[i % 3]:
        if img_path.exists():
            st.image(Image.open(img_path), caption=label, use_container_width=True)

st.write("")

# ------------------------
# About
# ------------------------
st.markdown("## 🌍 About this project")
st.markdown("""
<div class="about">
This app analyses a job-market dataset to show how industries differ in <b>job growth outlook</b>,
<b>salary</b>, <b>automation risk</b> and <b>AI adoption level</b>.
It has an industries page, an interactive dashboard with an industry filter, a comparison page
for automation risk and AI adoption, and an AI assistant chatbot for general questions about AI.
</div>
""", unsafe_allow_html=True)

st.write("")

# ------------------------
# Data source
# ------------------------
st.markdown("## 📂 Data Source")
st.info(
    "**Dataset:** AI-Powered Job Market Insights (Kaggle), 500 job listings across 10 industries.  \n"
    "**Important:** this is a synthetic dataset made for learning, so the results show my analysis "
    "skills and are not real-world statistics.  \n"
    "**Tools:** Python, Pandas, Plotly, Streamlit."
)

# ------------------------
# Explore
# ------------------------
st.write("")
st.markdown("## 🚀 Explore the project")

b1, b2, b3, b4 = st.columns(4)
b1.page_link("pages/2.AI Industries.py", label="Explore industries", icon="🏭")
b2.page_link("pages/3.AI future prediction.py", label="Future outlook", icon="🔮")
b3.page_link("pages/AI assistant.py", label="Ask the AI assistant", icon="🤖")
b4.page_link("pages/AI Dashboard.py", label="Open dashboard", icon="📊")

st.write("---")
st.caption("Developed by Sayali Dattatray Raktate")