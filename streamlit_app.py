import streamlit as st
import requests

st.set_page_config(page_title="Environmental Research Synthesizer", layout="centered")

# Status check
try:
    requests.get("http://localhost:8000/health", timeout=2)
    st.success("Backend is running")
except:
    st.error("Backend is not reachable")


API_URL = "http://localhost:8000/query"

st.title("🌱 Environmental Research Synthesizer")
st.markdown(
    "Ask an evidence-based research question. Answers are synthesized **only** from the underlying academic sources."
)

question = st.text_area(
    "Research question",
    placeholder="e.g. What are the social impacts of wind energy adoption?",
)

ask_button = st.button("Ask")

if ask_button and question.strip():
    with st.spinner("Retrieving evidence and synthesizing answer..."):
        try:
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            st.stop()

    if not data.get("in_scope", False):
        st.warning("⚠️ The question cannot be answered from the available sources.")
        if data.get("limitations"):
            st.markdown("**Reason:**")
            for lim in data["limitations"]:
                st.write(f"- {lim}")
        st.stop()

    st.subheader("🧠 Synthesized Answer")
    for bullet in data.get("answer", []):
        st.markdown(f"- {bullet['text']}")
        if bullet.get("citations"):
            st.caption("Citations: " + ", ".join(bullet["citations"]))

    if data.get("limitations"):
        st.subheader("⚠️ Limitations")
        for lim in data["limitations"]:
            st.write(f"- {lim}")

    if data.get("sources"):
        st.subheader("📚 Sources")
        for src in data["sources"]:
            author_year = f"{src['authors']} ({src['year']})"
            journal = f" — {src['journal']}" if src.get("journal") else ""
            st.markdown(
                f"<small>{author_year}{journal}</small><br>"
                f"<strong>**{src['title']}**</strong>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("⏱️ Metadata")
    st.json(data.get("meta", {}))

elif ask_button:
    st.warning("Please enter a question before clicking Ask.")