import streamlit as st
import requests

from utils.citations import CitationStyle
from utils.UI_rendering import render_confidence_profile, render_sentence_with_inline_citations, show_limitations, show_metadata, show_sources, show_query_expansion, show_trace
from utils.export import export_output
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
# st.title("Energy & Society Research Assistant")
st.markdown(
    "<h1 style='font-size: 2.6rem;'>Energy & Society Research Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown("""

Ask a research question about the social impacts of the energy transition.  
Answers are generated strictly from academic sources, and the system abstains when evidence is insufficient.
""")

# ---------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("💡 Try an example query", expanded=False):

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
    placeholder="Write your query here…",
)

topk_faiss = 30
topk_bm25 = 30

ask_button = st.button("Submit")

# ---------------------------------------------------------------------
# Trigger backend call (button ONLY sets state)
# ---------------------------------------------------------------------
answer_placeholder = st.empty()

if ask_button:
    if not question.strip():
        st.warning("Please enter a question before clicking Submit.")
    
    else:
        # answer_placeholder.empty()  

        with st.spinner("Retrieving evidence and synthesizing answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "topk_faiss": topk_faiss, "topk_bm25": topk_bm25},
                    timeout=60,
                )
                st.session_state["data"] = response.json()
            except requests.exceptions.Timeout:
                st.error("⏳ The request timed out. Please try again.")
                st.stop()
            except Exception as e:
                st.error(f"Backend request failed. Please try again.")
                st.stop()

data = st.session_state.get("data")
if data:
    with answer_placeholder.container():
        st.markdown("---")

        # ---------------------------------------------------------------------
        # Pipeline failures handling
        # ---------------------------------------------------------------------
        pipeline_status = data.get("pipeline_status", "")

        if pipeline_status == "out_of_scope":
            show_limitations(data, level="warning")
            show_metadata(data)
            st.stop()

        if pipeline_status != "success":
            show_limitations(data, level="error")
            show_metadata(data)
            st.stop()
        
        # ---------------------------------------------------------------------
        # Early returns
        # ---------------------------------------------------------------------
        confidence = data["confidence"]

        if confidence["status"] == "Not applicable":
            if confidence["reason"] == "Absent evidence":
                show_limitations(data, case="absent_evidence")
            elif confidence["reason"] == "Abstention":
                show_limitations(data, case="abstention")
            show_metadata(data)
            show_query_expansion(data)
            show_trace(data)
            st.stop()

        # ---------------------------------------------------------------------
        # Synthesized answer (inline citations)
        # ---------------------------------------------------------------------
        render_confidence_profile(confidence)
        
        citation_style = CitationStyle.NUMERIC

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Synthesized Answer")

        for item in data.get("answer", []):
            st.markdown(render_sentence_with_inline_citations(item, citation_style))

        # ---------------------------------------------------------------------
        # Limitations
        # ---------------------------------------------------------------------
        if data.get("limitations"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Limitations")
            show_limitations(data)

        # ---------------------------------------------------------------------
        # Sources (paper-level bibliography)
        # ---------------------------------------------------------------------
        if data.get("sources"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Sources")
            show_sources(data, citation_style)
            
        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------
        show_metadata(data)

        # ---------------------------------------------------------------------
        # Query expansion
        # ---------------------------------------------------------------------
        show_query_expansion(data)       

        # ---------------------------------------------------------------------
        # Trace panel (sidebar-controlled)
        # ---------------------------------------------------------------------
        show_trace(data)

        st.markdown("---")
        st.subheader("Export")
        export_output(data)