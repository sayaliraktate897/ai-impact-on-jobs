import streamlit as st

# ------------------------
# Sidebar navigation (grouped, clean names, icons)
# ------------------------

pages = {
    "Overview": [
        st.Page("pages/1.Home.py", title="Home", icon="🏠", default=True),
    ],
    "Explore": [
        st.Page("pages/2.AI Industries.py", title="AI industries", icon="🏭"),
        st.Page("pages/3.AI future prediction.py", title="Future prediction", icon="🔮"),
    ],
    "Tools": [
        st.Page("pages/AI assistant.py", title="AI assistant", icon="🤖"),
        st.Page("pages/AI Dashboard.py", title="AI dashboard", icon="📊"),
    ],
}

pg = st.navigation(pages)

# Run the selected page first, so this footer appears below its sidebar content
pg.run()

# ------------------------
# Sidebar footer (shown on every page)
# ------------------------

st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Impact on Jobs")

st.sidebar.caption("Exploring AI, jobs and automation risk")
st.sidebar.caption("Built with Python and Streamlit")