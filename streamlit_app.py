import streamlit as st
import requests

from utils.inline_citations import render_sentence_with_inline_citations

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Environmental Research Synthesizer",
    layout="centered"
)

# ---------------------------------------------------------------------
# Backend health check
# ---------------------------------------------------------------------
API_URL = "http://localhost:8000/query"
HEALTH_URL = "http://localhost:8000/health"

try:
    requests.get(HEALTH_URL, timeout=2)
    st.success("Backend is running")
except:
    st.error("Backend is not reachable")

# ---------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------
st.title("🌱 Environmental Research Synthesizer")
st.markdown(
    "Ask an evidence-based research question. "
    "Answers are synthesized **only** from the underlying academic sources."
)

# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------
question = st.text_area(
    "Research question",
    placeholder="e.g. What are the social impacts of wind energy adoption?",
)

ask_button = st.button("Ask")

# ---------------------------------------------------------------------
# Trigger backend call (button ONLY sets state)
# ---------------------------------------------------------------------
if ask_button:
    if not question.strip():
        st.warning("Please enter a question before clicking Ask.")
    else:
        with st.spinner("Retrieving evidence and synthesizing answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=60,
                )
                response.raise_for_status()
                st.session_state["data"] = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                st.stop()

# ---------------------------------------------------------------------
# Read state (ALL rendering depends on this)
# ---------------------------------------------------------------------
data = st.session_state.get("data")
if not data:
    st.stop()

# ---------------------------------------------------------------------
# Reason handling
# ---------------------------------------------------------------------
reason = data.get("reason", "none")

if reason == "out_of_scope":
    st.warning("⚠️ The question cannot be answered from the available sources.")
    if data.get("limitations"):
        st.markdown("**Reason:**")
        for lim in data["limitations"]:
            st.write(f"- {lim}")
    st.stop()

elif reason == "insufficient_evidence":
    st.warning(
        "⚠️ The available evidence is limited. "
        "The answer below reflects only what is directly supported by the sources."
    )

# ---------------------------------------------------------------------
# Synthesized answer (inline citations)
# ---------------------------------------------------------------------
st.subheader("🧠 Synthesized Answer")

for item in data.get("answer", []):
    st.markdown(render_sentence_with_inline_citations(item))

# ---------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------
if data.get("limitations"):
    st.subheader("⚠️ Limitations")
    for lim in data["limitations"]:
        st.write(f"- {lim}")

# ---------------------------------------------------------------------
# Sources (paper-level bibliography)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------
st.subheader("⏱️ Metadata")
st.json(data.get("meta", {}))

# ---------------------------------------------------------------------
# Debug panel (sidebar-controlled)
# ---------------------------------------------------------------------
with st.sidebar:
    show_debug = st.checkbox("🧪 Show debug panel", value=False)

if show_debug and "debug" in data:
    st.subheader("🧪 Debug: Evidence Trace")

    debug = data["debug"]

    col1, col2 = st.columns(2)

    # Retrieved chunks
    with col1:
        st.markdown("### 🔎 Retrieved Chunks")
        for c in debug["retrieved_chunks"]:
            st.code(
                f"{c['chunk_id']}\n"
                f"{c['title']} ({c['year']})\n\n"
                f"{c['text'][:300]}..."
            )

    # Used chunks + metrics
    with col2:
        st.markdown("### ✅ Used in Synthesis")

        used_ids = {c["chunk_id"] for c in debug["used_chunks"]}

        for c in debug["retrieved_chunks"]:
            if c["chunk_id"] in used_ids:
                st.code(
                    f"{c['chunk_id']}\n"
                    f"{c['title']} ({c['year']})\n\n"
                    f"{c['text'][:300]}..."
                )

        metrics = debug["retrieval_metrics"]

        st.markdown("### 📊 Evidence Metrics")
        st.metric("Retrieved chunks", metrics["retrieved"])
        st.metric("Used chunks", metrics["used"])
        st.metric("Coverage", f"{metrics['coverage']:.0%}")
        st.metric("Unique papers used", metrics["unique_papers_used"])
