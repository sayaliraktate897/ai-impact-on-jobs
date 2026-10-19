import os
import re
import time
import pandas as pd
import streamlit as st
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 AI Impact on Jobs Chatbot")
st.caption("Ask about AI, or ask anything about the job-market dataset in this project. Every answer shows its source.")


# ---------- Load the dataset ----------
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, "data", "archive (9)", "ai_job_market_insights.csv"))


df = load_data()
n_jobs = len(df)

summary = df.groupby("Industry").agg(
    Salary=("Salary_USD", "mean"),
    Growth=("Job_Growth_Projection", lambda x: (x == "Growth").mean() * 100),
    Risk=("Automation_Risk", lambda x: (x == "High").mean() * 100),
    Remote=("Remote_Friendly", lambda x: (x == "Yes").mean() * 100),
).round(1)

INDUSTRIES = list(summary.index)
TITLES = sorted(df["Job_Title"].astype(str).unique()) if "Job_Title" in df.columns else []
LOCATIONS = sorted(df["Location"].astype(str).unique()) if "Location" in df.columns else []
ENTITIES = [e.lower() for e in INDUSTRIES + TITLES + LOCATIONS]


# ---------- Data summary text for the LLM (used for any dataset question) ----------
def pct_table(data, col):
    g = data.groupby(col)
    t = pd.DataFrame({"jobs": g.size(), "avg_salary": g["Salary_USD"].mean().round(0)})
    for target in ["Job_Growth_Projection", "Automation_Risk", "AI_Adoption_Level", "Remote_Friendly"]:
        if target in data.columns and target != col:
            for v in sorted(data[target].astype(str).unique()):
                share = (data[target].astype(str) == v).groupby(data[col]).mean() * 100
                t[f"{target}={v} (%)"] = share.round(1)
    return t.reset_index().to_csv(index=False)


@st.cache_data
def build_context(data):
    parts = [
        f"The dataset has {len(data)} job listings. Columns: {', '.join(data.columns)}.",
        "It is a synthetic Kaggle dataset (AI-Powered Job Market Insights). It has no year column.",
    ]
    for col in ["Industry", "Job_Title", "Company_Size", "Location", "AI_Adoption_Level",
                "Automation_Risk", "Remote_Friendly", "Job_Growth_Projection"]:
        if col in data.columns:
            parts.append(f"Values of {col}: {', '.join(sorted(data[col].astype(str).unique()))}.")
    s = data["Salary_USD"]
    parts.append(f"Salary_USD: min {s.min():,.0f}, max {s.max():,.0f}, mean {s.mean():,.0f}, median {s.median():,.0f}.")

    for col in ["Industry", "Job_Title", "Company_Size", "Location"]:
        if col in data.columns:
            parts.append(f"TABLE by {col} (jobs = number of listings, avg_salary in USD, other columns are % of listings):\n"
                         + pct_table(data, col))

    if {"Industry", "Job_Title"}.issubset(data.columns):
        g = data.groupby(["Industry", "Job_Title"])
        t = pd.DataFrame({"jobs": g.size(), "avg_salary": g["Salary_USD"].mean().round(0)}).reset_index()
        parts.append("TABLE by Industry and Job_Title:\n" + t.to_csv(index=False))

    for col in ["Automation_Risk", "AI_Adoption_Level", "Job_Growth_Projection", "Remote_Friendly"]:
        if col in data.columns:
            avg = data.groupby(col)["Salary_USD"].mean().round(0)
            parts.append(f"Average salary by {col}: " + ", ".join(f"{k} {v:,.0f}" for k, v in avg.items()) + ".")

    if {"Automation_Risk", "AI_Adoption_Level"}.issubset(data.columns):
        parts.append("Number of jobs by Automation_Risk (rows) and AI_Adoption_Level (columns):\n"
                     + pd.crosstab(data["Automation_Risk"], data["AI_Adoption_Level"]).to_csv())

    if "Required_Skills" in data.columns:
        try:
            skills = data["Required_Skills"].dropna().astype(str).str.split(r"[;,|]").explode().str.strip()
            top = skills.value_counts().head(15)
            parts.append("Most common Required_Skills (count): "
                         + ", ".join(f"{k} {v}" for k, v in top.items()) + ".")
        except Exception:
            pass
    return "\n\n".join(parts)


# ---------- Exact answers calculated with Pandas (simple ranking questions) ----------
METRICS = {
    "salary": ("Salary", "average salary", lambda v: f"USD {v:,.0f}"),
    "risk": ("Risk", "share of high automation risk jobs", lambda v: f"{v:.1f}%"),
    "growth": ("Growth", "share of jobs with a growth outlook", lambda v: f"{v:.1f}%"),
    "remote": ("Remote", "share of remote-friendly jobs", lambda v: f"{v:.1f}%"),
}
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}


def detect_metric(t):
    if re.search(r"\b(salary|salaries|pay|paid|earn|earns|earning|earnings)\b", t):
        return "salary"
    if re.search(r"\b(automat\w*|risk\w*)\b", t):
        return "risk"
    if re.search(r"\bgrow\w*", t):
        return "growth"
    if re.search(r"\bremote\b", t):
        return "remote"
    return None


def count_metrics(t):
    patterns = [r"\b(salary|salaries|pay|paid|earn|earns|earning|earnings)\b",
                r"\b(automat\w*|risk\w*)\b", r"\bgrow\w*", r"\bremote\b"]
    return sum(1 for p in patterns if re.search(p, t))


def top_n(t):
    m = re.search(r"\b(\d+)\b", t)
    if m:
        return int(m.group(1))
    for w, n in NUMBER_WORDS.items():
        if w in t.split():
            return n
    return None


def data_answer(text):
    t = text.lower()
    if "how many" in t and ("job" in t or "listing" in t) and not any(e in t for e in ENTITIES):
        return f"The dataset has {n_jobs} job listings across {len(summary)} industries."
    if "how many" in t and "industr" in t:
        return f"The dataset covers {len(summary)} industries."

    metric = detect_metric(t)
    if not metric:
        return None
    # questions about job titles or locations need the full data summary
    if any(x.lower() in t for x in TITLES + LOCATIONS):
        return None

    found = [i for i in INDUSTRIES if i.lower() in t]
    if len(found) > 1 or "compare" in t or count_metrics(t) > 1:
        return None  # comparisons and multi-part questions go to the LLM with the data summary
    industry = found[0] if found else None
    ranking = any(w in t for w in ("top", "bottom", "best", "worst", "highest", "lowest",
                                   "most", "least", "rank"))
    if industry is None and not ranking:
        return None

    col, label, fmt = METRICS[metric]
    if industry:
        value = summary.loc[industry, col]
        rank = int(summary[col].rank(ascending=False)[industry])
        return (f"In the dataset, the {label} for {industry} is {fmt(value)}. "
                f"It ranks {rank} out of {len(summary)} industries.")

    low = any(w in t for w in ("bottom", "lowest", "least", "worst"))
    n = top_n(t)
    if n is None and ("top" in t or "bottom" in t):
        n = 3
    if n and n > 1:
        n = min(n, len(summary))
        ordered = summary[col].sort_values(ascending=low)
        lines = [f"{i + 1}. {name}: {fmt(v)}" for i, (name, v) in enumerate(ordered.head(n).items())]
        word = "lowest" if low else "highest"
        return f"In the dataset, the {n} industries with the {word} {label} are:\n\n" + "\n".join(lines)

    ordered = summary[col].sort_values(ascending=False)
    return (f"In the dataset, {ordered.index[0]} has the highest {label} ({fmt(ordered.iloc[0])}), "
            f"and {ordered.index[-1]} has the lowest ({fmt(ordered.iloc[-1])}).")


# ---------- Is the question about the project dataset? ----------
EXPLICIT = ("dataset", "data set", "my data", "in the data", "my project", "this project",
            "job listing", "listings")
METRIC_RE = re.compile(r"\b(salary|salaries|pay|earn\w*|remote|risk\w*|automat\w*|adoption|growth|grow\w*|"
                       r"average|highest|lowest|most|least|top|bottom|compare|share|percent\w*|count|number)\b")
ENTITY_RE = re.compile(r"\b(industry|industries|job titles?|locations?|company sizes?)\b")


def is_data_question(text):
    t = text.lower()
    if any(w in t for w in EXPLICIT):
        return True
    entity = any(e in t for e in ENTITIES) or ENTITY_RE.search(t)
    metric = METRIC_RE.search(t) or "how many" in t
    return bool(entity and metric)


# ---------- Knowledge base (FAQ) ----------
faq = {
    "What is AI?": "AI is the ability of machines to perform tasks that normally need human intelligence.",
    "What is machine learning?": "Machine learning lets computers learn patterns from data without being explicitly programmed.",
    "What is deep learning?": "Deep learning uses multi-layer neural networks to learn complex patterns.",
    "What is NLP?": "NLP means Natural Language Processing. It helps computers understand and work with human language.",
    "What is an LLM?": "An LLM is a Large Language Model, an AI trained on a huge amount of text so it can understand and write language. Gemini is an LLM.",
    "What is ChatGPT?": "ChatGPT is a large language model that generates human-like text.",
    "What is the future of AI?": "AI is expected to transform healthcare, education and finance in the coming years.",
    "How is AI used in healthcare?": "AI helps in disease detection, medical imaging and drug discovery.",
    "What are AI jobs?": "Data scientist, ML engineer, AI researcher and data analyst are popular AI careers.",
    "Will AI replace humans?": "AI will automate some tasks but also create new jobs.",
    "What is this project about?": "AI Impact on Jobs analyses a job-market dataset to compare industries on job growth, salary and automation risk.",
    "What data does this project use?": "It uses a Kaggle dataset, AI-Powered Job Market Insights, with 500 synthetic job listings across 10 industries.",
}
questions, answers = list(faq.keys()), list(faq.values())
GREETINGS = {"hi", "hii", "hello", "hey", "heyy", "thanks", "thank", "ok", "okay"}


def preprocess(text):
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return " ".join(w for w in text.split() if w not in GREETINGS)


@st.cache_resource
def build_vectors():
    vec = TfidfVectorizer(stop_words="english")
    qv = vec.fit_transform([preprocess(q) for q in questions])
    return vec, qv


vectorizer, question_vectors = build_vectors()


def nlp_answer(user_input):
    clean = preprocess(user_input)
    vec = vectorizer.transform([clean])
    scores = cosine_similarity(vec, question_vectors)[0]
    best = scores.argmax()
    words = vectorizer.build_analyzer()(clean)
    known = [w for w in words if w in vectorizer.vocabulary_]
    coverage = len(known) / len(words) if words else 0
    return answers[best], scores[best], coverage


# ---------- Gemini API ----------
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash"]


def call_gemini(prompt):
    client = genai.Client(api_key=st.secrets["API_KEY"])
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text.replace("$", "\\$")
            except Exception as e:
                last_error = e
                msg = str(e)
                if "NOT_FOUND" in msg or "404" in msg:
                    break  # this model is not available, try the next model
                time.sleep(2)
    raise last_error


def general_answer(user_input):
    return call_gemini(
        "You are a helpful assistant for a student project about how AI affects jobs and industries. "
        "Answer in simple English, in at most 120 words.\n\nQuestion: " + user_input
    )


def dataset_llm_answer(user_input):
    return call_gemini(
        "You answer questions about a job-market dataset used in a student project. "
        "Use ONLY the data below. Do not guess and do not use outside knowledge. "
        "If the data below cannot answer the question, say that the dataset does not show it. "
        "Answer in simple English, in at most 100 words, and include the numbers.\n\n"
        "DATA:\n" + build_context(df) + "\n\nQuestion: " + user_input
    )


def api_error_message(e):
    msg = str(e)
    if "NOT_FOUND" in msg or "404" in msg:
        return ("Sorry, the AI model is not available right now. "
                "I can still answer simple dataset questions and FAQ questions.")
    return ("Sorry, the AI service is busy right now. Please try again in a minute. "
            "I can still answer simple dataset questions and FAQ questions.")


# ---------- Reply logic ----------
def get_reply(user_input):
    words = preprocess(user_input).split()
    if not words:
        return "Hello! Ask me about AI, or ask anything about the job-market dataset in this project."

    answer = data_answer(user_input)
    if answer:
        return f"{answer}\n\n*(Source: project dataset)*"

    faq_text, score, coverage = nlp_answer(user_input)
    if score >= 0.6 and coverage >= 0.75:
        return f"{faq_text}\n\n*(Source: knowledge base, similarity {score:.2f})*"

    if is_data_question(user_input):
        try:
            return dataset_llm_answer(user_input) + "\n\n*(Source: project dataset, explained by LLM)*"
        except Exception as e:
            return api_error_message(e)

    if score >= 0.3 and coverage >= 0.7:
        return f"{faq_text}\n\n*(Source: knowledge base, similarity {score:.2f})*"

    try:
        return general_answer(user_input) + "\n\n*(Source: LLM API)*"
    except Exception as e:
        return api_error_message(e)


# ---------- Chat UI ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

top1, top2 = st.columns([4, 1])
with top1:
    with st.expander("💡 Try these questions"):
        st.markdown(
            "- What is machine learning? *(knowledge base)*\n"
            "- Which industry has the highest salary? *(project dataset)*\n"
            "- Top 3 industries by automation risk *(project dataset)*\n"
            "- Compare remote jobs and salary in Finance and Retail *(dataset, explained by LLM)*\n"
            "- How will AI change finance jobs? *(LLM API)*"
        )
with top2:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything about AI or about the project data...")
if prompt and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_reply(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})