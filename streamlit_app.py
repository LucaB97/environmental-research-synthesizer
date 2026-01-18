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
# st.subtitle("🌱 Environmental Research Synthesizer")
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

top_k = st.slider(
    "Chunks to retrieve",
    min_value=5,
    max_value=30,
    value=15,
    step=5
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
                    json={
                        "question": question,
                        "top_k": top_k
                    },
                    timeout=60,
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⏳ The request timed out. The system may be under heavy load. "
                    "Please try again."
                )
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(
                    "🚫 Unable to contact the backend service. "
                    "Please check that it is running."
                )
                st.stop()

            # ---- HTTP-level failure (true backend error) ----
            if response.status_code != 200:
                st.error(
                    "🚫 The backend service encountered an error. "
                    "Please try again later."
                )
                st.stop()

            # ---- Success: parse JSON ----
            try:
                st.session_state["data"] = response.json()
            except ValueError:
                st.error(
                    "🚫 Received an invalid response from the backend."
                )
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

if reason == "generation_failed":
    st.warning("⚠️ Unable to generate a reliable answer.")
    st.markdown(
        "The system could not generate a stable synthesis from the available evidence."
    )
    st.stop()

elif reason == "out_of_scope":
    st.warning("⚠️ The question cannot be answered from the available sources.")
    if data.get("limitations"):
        st.markdown("**Reason:**")
        for lim in data["limitations"]:
            st.write(f"{lim}")
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
        st.write(f"{lim}")

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
    chunks = debug["chunks"]
    papers = debug["papers"]
    metrics = debug["metrics"]

    # -----------------------------------------------------------------
    # Evidence trace (grouped by paper)
    # -----------------------------------------------------------------
    for paper in papers:
        paper_id = paper["paper_id"]
        paper_chunks = [
            c for c in chunks if c["paper_id"] == paper_id
        ]

        used_chunks = [c for c in paper_chunks if c["used_in_synthesis"]]

        # Paper header
        # st.markdown(
        #     f"**📄 {paper['title']} ({paper['year']})**\n"
        #     f"*{paper['authors']}*  \n"
        #     f"Used chunks: **{paper['chunks_used']} / {paper['chunks_retrieved']}*"
        # )

        st.markdown(f"**📄 {paper['title']} ({paper['year']})**")
        st.caption(
            f"{paper['authors']} · "
            f"Used {paper['chunks_used']} / {paper['chunks_retrieved']} chunks"
        )


        # Chunk list
        for c in paper_chunks:
            is_used = c["used_in_synthesis"]

            border_color = "🟢" if is_used else "⚪"
            opacity = 1.0 if is_used else 0.6

            with st.expander(
                f"{border_color} {c['chunk_id'].split('__')[-1]} "
                f"(rank #{c['rank']})",
                expanded=False
            ):
                st.markdown(
                    f"<div style='opacity:{opacity}'>"
                    f"{c['text']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # -----------------------------------------------------------------
    # Evidence metrics
    # -----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Evidence Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Retrieved chunks", metrics["retrieved_chunks"])
        st.metric("Used chunks", metrics["used_chunks"])
        st.metric("Chunk coverage", f"{metrics['chunk_coverage']:.0%}")

    with col2:
        st.metric("Retrieved papers", metrics["retrieved_papers"])
        st.metric("Unique papers used", metrics["used_papers"])
        st.metric("Paper coverage", f"{metrics['paper_coverage']:.0%}")

    with col3:
        st.metric("Paper dominance", metrics["paper_dominance"])
        st.metric("Avg citations per sentence", metrics["avg_citations_per_sentence"])
        st.metric("Multi-source sentence ratio", metrics["multi_source_sentence_ratio"])
