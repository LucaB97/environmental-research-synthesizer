import streamlit as st
import requests

from utils.inline_citations import render_sentence_with_inline_citations
from utils.export import response_to_json, response_to_markdown
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
except:
    st.error("🚫 Backend is not reachable. ")
    st.stop()

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
# st.subheader("🧠 Synthesized Answer")

label = data["confidence"]["label"]
score = data["confidence"]["score"]
explanation = data["confidence"]["explanation"]

if label == "High":
    st.success(f"Confidence: {label}")
elif label == "Medium":
    st.warning(f"Confidence: {label}")
else:
    st.error(f"Confidence: {label}")

st.caption(
    "Confidence reflects how well the answer is supported by multiple independent sources."
)

with st.expander("Why this confidence level?"):
    for item in explanation:
        st.markdown(f"- {item}")

st.subheader("🧠 Synthesized Answer")
st.markdown("### Synthesized Answer")

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
            f"<div style='margin-bottom:8px;'>"
            f"<small>{author_year}{journal}</small><br>"
            f"<strong>{src['title']}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("⏱️ Metadata", expanded=False):
    st.json(data.get("meta", {}))

# ---------------------------------------------------------------------
# 📊 Evidence Metrics
# ---------------------------------------------------------------------
metrics = data.get("evidence_metrics")

if metrics is not None:
    with st.expander("📊 Evidence Metrics", expanded=False):
        
        if not metrics:
            st.info("Evidence metrics are unavailable for this response.")
        
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Retrieved chunks", metrics.get('retrieved_chunks', 0))
                st.metric("Used chunks", metrics.get('used_chunks', 0))
                st.metric(
                    "Chunk coverage",
                    f"{metrics.get('chunk_coverage', 0):.0%}"
                )

            with col2:
                st.metric("Retrieved papers", metrics.get('retrieved_papers', 0))
                st.metric("Unique papers used", metrics.get('used_papers', 0))
                st.metric(
                    "Paper coverage",
                    f"{metrics.get('paper_coverage', 0):.0%}"
                )

            with col3:
                st.metric("Paper dominance", metrics.get('paper_dominance', 0))
                st.metric(
                    "Avg citations / sentence",
                    metrics.get('avg_citations_per_sentence', 0)
                )
                st.metric(
                    "Multi-source sentences",
                    f"{metrics.get('multi_source_sentence_ratio', 0):.0%}"
                )

# ---------------------------------------------------------------------
# Debug panel (sidebar-controlled)
# ---------------------------------------------------------------------
debug = data.get("debug")

with st.sidebar:
    show_debug = st.checkbox("🧪 Show debug panel", value=False)

if show_debug and debug:
    chunks = debug.get("chunks", [])
    papers = debug.get("papers", [])       

    # =============================================================
    # 📄 Evidence Trace
    # =============================================================

    if papers and chunks:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📄 Evidence Trace")

        for paper in papers:
            paper_id = paper["paper_id"]

            paper_chunks = [
                c for c in chunks
                if c["paper_id"] == paper_id
            ]

            used_chunks = [
                c for c in paper_chunks
                if c["used_in_synthesis"]
            ]

            # Paper header
            st.markdown(
                f"**📄 {paper['title']} ({paper['year']})**"
            )
            st.caption(
                f"{paper['authors']} · "
                f"Used {paper['chunks_used']} / "
                f"{paper['chunks_retrieved']} chunks"
            )

            # Chunk list
            for c in paper_chunks:
                is_used = c["used_in_synthesis"]

                indicator = "🟢" if is_used else "⚪"
                opacity = 1.0 if is_used else 0.6

                with st.expander(
                    f"{indicator} "
                    f"{c['chunk_id'].split('__')[-1]} "
                    f"(rank #{c['rank']})",
                    expanded=False
                ):
                    st.markdown(
                        f"<div style='opacity:{opacity}'>"
                        f"{c['text']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        
    else:
        st.info("No debug information available for this response.")

# with st.expander("Export results"):
#     export_format = st.radio(
#         "Choose format",
#         options=["JSON", "Markdown"],
#         horizontal=True,
#     )

#     if export_format == "JSON":
#         data_export = response_to_json(data)
#         filename = "query_response.json"
#         mime = "application/json"
#     else:
#         data_export = response_to_markdown(data)
#         filename = "query_response.md"
#         mime = "text/markdown"

#     st.download_button(
#         label="Download",
#         data=data_export,
#         file_name=filename,
#         mime=mime,
#     )

st.markdown("---")
st.markdown("### ⬇️ Export")

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    with st.container():
        export_format = st.radio(
            "Format",
            ["JSON", "Markdown"],
            horizontal=True,
            label_visibility="collapsed"
        )
            
        if export_format == "JSON":
            has_debug = bool(data.get("debug"))
            include_debug = st.checkbox(
                "Include debug evidence (chunks & papers)",
                value=False,
                disabled=not has_debug
            )
            data_export = response_to_json(data, include_debug)
            filename = "query_response.json"
            mime = "application/json"

        else:
            data_export = response_to_markdown(data)
            filename = "query_response.md"
            mime = "text/markdown"

        st.download_button(
            label="⬇️ Download results",
            data=data_export,
            file_name=filename,
            mime=mime,
            use_container_width=True,
        )


