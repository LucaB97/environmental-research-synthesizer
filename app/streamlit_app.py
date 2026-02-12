import streamlit as st
import requests

from utils.citations import CitationStyle, render_sentence_with_inline_citations
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
st.markdown(
    "Ask an evidence-based research question. "
    "Answers are synthesized **only** from the underlying academic sources."
)


# ---------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("💡 Try an example query:")

example_queries = [
    "What are the costs and benefits of renewable energy adoption?",
    "How do ownership models shape social outcomes of renewable projects?",
    "What is the impact of energy transition on the labour market?",
    "How do political views affect the opinions on renewable energy?",
    "What is the societal response to renewable energy projects?",
    "Which social groups are most affected by renewable energy projects?"
]

row1 = example_queries[:3]
row2 = example_queries[3:]

cols1 = st.columns(3)
for i, ex in enumerate(row1):
    if cols1[i].button(ex):
        st.session_state["question"] = ex
        if 'answer_placeholder' in st.session_state:
            st.session_state['answer_placeholder'].empty()

cols2 = st.columns(3)
for i, ex in enumerate(row2):
    if cols2[i].button(ex):
        st.session_state["question"] = ex
        if 'answer_placeholder' in st.session_state:
            st.session_state['answer_placeholder'].empty()

st.markdown('</div>', unsafe_allow_html=True)
# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------
if 'question' not in st.session_state:
    st.session_state['question'] = ""

question = st.text_area(
    "Research question",
    value=st.session_state['question'],
    placeholder="Write your question here…",
)

# top_k = st.slider(
#     "Chunks to retrieve",
#     min_value=10,
#     max_value=30,
#     value=20,
#     step=5
# )

top_k_faiss = 30
top_k_bm25 = 30

ask_button = st.button("Ask")

# ---------------------------------------------------------------------
# Trigger backend call (button ONLY sets state)
# ---------------------------------------------------------------------
answer_placeholder = st.empty()

if ask_button:
    if not question.strip():
        st.warning("Please enter a question before clicking Ask.")
    
    else:
        answer_placeholder.empty()  

        with st.spinner("Retrieving evidence and synthesizing answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "top_k_faiss": top_k_faiss, "top_k_bm25": top_k_bm25},
                    timeout=60,
                )
                st.session_state["data"] = response.json()
            except Exception as e:
                st.error(f"Backend request failed: {e}")
                st.stop()
            
            # try:
            #     response = requests.post(
            #         API_URL,
            #         json={
            #             "question": question,
            #             "top_k": top_k
            #         },
            #         timeout=60,
            #     )
            # except requests.exceptions.Timeout:
            #     st.error(
            #         "⏳ The request timed out. "
            #         "Please try again."
            #     )
            #     st.stop()
            # except requests.exceptions.RequestException as e:
            #     st.error(
            #         "🚫 Unable to contact the backend service. "
            #         "Please check that it is running."
            #     )
            #     st.stop()

            # # ---- HTTP-level failure (true backend error) ----
            # if response.status_code != 200:
            #     st.error(
            #         "🚫 The backend service encountered an error. "
            #         "Please try again later."
            #     )
            #     st.stop()

            # # ---- Success: parse JSON ----
            # try:
            #     st.session_state["data"] = response.json()
            # except ValueError:
            #     st.error(
            #         "🚫 Received an invalid response from the backend."
            #     )
            #     st.stop()


# Once data is ready, render inside the same placeholder
with answer_placeholder.container():
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
        st.warning("The question cannot be answered from the available sources.")
        st.stop()


    elif reason == "retrieval_empty":
        st.warning("No documents could be retrieved for this question.")
        st.stop()


    elif reason == "absent_evidence":
        st.warning("The literature retrieved is topically related, but does not address this question directly.")
        if data.get("meta", {}):
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Metadata", expanded=False):
                st.json(data.get("meta", {}))
        st.stop()


    elif reason == "isolated_evidence":
        st.warning("The retrieved evidence is too narrow and context-specific to support synthesis across studies.")
        if data.get("meta", {}):
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Metadata", expanded=False):
                st.json(data.get("meta", {}))

        debug = data.get("debug")

        with st.sidebar:
            show_debug = st.checkbox("Show relevant evidence", value=False)
        
        if show_debug and debug:
            chunks = debug.get("chunks", [])
        
            if chunks:
                st.markdown("<br>", unsafe_allow_html=True)
                for chunk in chunks:
                    st.markdown("---")
                    st.markdown(f"**📄 {chunk['title']}**")
                    st.caption(f"{chunk['authors']} ({chunk['year']})")
                    st.text_area("Excerpt", chunk['text'], height=120)
            else:
                st.info("No information available.")
        st.stop()


    elif reason == "insufficient_evidence":
        st.warning(
            "⚠️ The available evidence is limited."
        )

        # Case 1: partial answer → explain and continue
        if data.get("answer", []):
            st.info(
                "The answer below reflects only what is directly supported by the sources."
            )
        
        # Case 2: no answer at all → explain and stop
        else:
            limitations = data.get("limitations", [])
            if limitations:
                for lim in limitations:
                    st.info(lim)
            else:
                st.info(
                    "No meaningful answer could be produced from the available literature."
                )
            
            if data.get("meta", {}):
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Metadata", expanded=False):
                    st.json(data.get("meta", {}))
                    
            st.stop()       


    elif reason == "generation_failed":
        st.error(
        "The system could not generate a reliable answer this time. "
        "Please try again."
        )
        st.stop()


    # ---------------------------------------------------------------------
    # Confidence level
    # ---------------------------------------------------------------------

    if reason != "insufficient_evidence":
        
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
            if explanation:
                for item in explanation:
                    st.markdown(f"- {item}")
            else:
                st.markdown(
                    "No specific confidence drivers were triggered. "
                    "This confidence level reflects an overall assessment of the available evidence."
                )


    # ---------------------------------------------------------------------
    # Synthesized answer (inline citations)
    # ---------------------------------------------------------------------
    citation_style = CitationStyle.NUMERIC

    st.subheader("Synthesized Answer")

    for item in data.get("answer", []):
        st.markdown(render_sentence_with_inline_citations(item, citation_style))

    # ---------------------------------------------------------------------
    # Limitations
    # ---------------------------------------------------------------------
    if data.get("limitations"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Limitations")
        for lim in data["limitations"]:
            st.write(f"{lim}")

    # ---------------------------------------------------------------------
    # Sources (paper-level bibliography)
    # ---------------------------------------------------------------------
    if data.get("sources"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Sources")

        for src in data["sources"]:
            authors = src["authors"]
            year = src["year"]
            title = src["title"]
            journal = f" — {src['journal']}" if src.get("journal") else ""

            if citation_style == CitationStyle.NUMERIC:
                number = src.get("citation_number")
                prefix = f"[{number}] " if number is not None else ""
            else:
                prefix = ""

            st.markdown(
                f"<div style='margin-bottom:16px;'>"
                f"<small>{prefix}{authors} ({year}){journal}</small><br>"
                f"<strong>{title}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Metadata", expanded=False):
        st.json(data.get("meta", {}))

    # ---------------------------------------------------------------------
    # 📊 Evidence Metrics
    # ---------------------------------------------------------------------
    metrics = data.get("evidence_metrics")

    if metrics is not None:
        with st.expander("Evidence Metrics", expanded=False):
            
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
                    st.metric("Paper dominance", metrics.get('paper_dominance', 0))

                with col3:
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
        show_debug = st.checkbox("Show debug panel", value=False)

    if show_debug and debug:
        chunks = debug.get("chunks", [])
        papers = debug.get("papers", [])       

        # =============================================================
        # 📄 Evidence Trace
        # =============================================================

        if papers and chunks:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Evidence Trace")

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


    st.markdown("---")
    st.subheader("Export")

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
                label="Download results",
                data=data_export,
                file_name=filename,
                mime=mime,
                use_container_width=True,
            )