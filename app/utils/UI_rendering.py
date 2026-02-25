import streamlit as st
from utils.citations import CitationStyle


### Confidence profile

def render_confidence_profile(confidence):

    st.subheader("Confidence Profile")
    st.caption(
        "The Confidence Profile evaluates (1) the strength and distribution of retrieved evidence "
        "and (2) how well the answer integrates and balances the cited sources. "
        "It does not assess factual correctness."
    )
    
    evidence = confidence["evidence"]
    grounding = confidence["grounding"]

    level_colors = {
        "Strong": "#007acc",      # calm blue
        "Moderate": "#a17fcf",    # soft purple
        "Weak": "#888888",        # neutral gray
        "Not_applicable": "#aaaaaa"
    }

    st.markdown("""
    <style>
    .confidence-metric {
        margin-bottom: 1.2rem;
    }

    .metric-label {
        font-size: 0.875rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-size: 2.1rem;
        font-weight: 600;
        line-height: 1.1;
        margin: 0;
    }

    .confidence-level {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    # ---- Evidence ----
    with col1:
        st.markdown(f"""
        <div class="confidence-metric">
            <div class="metric-label">Evidence structure</div>
            <div class="metric-value">{evidence['score']:.2f}</div>
            <div class="confidence-level"
                 style="color:{level_colors[evidence['level']]}">
                 {evidence['level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Why this score?"):
            if evidence["explanation"]["weaknesses"]:
                st.markdown("**Weaknesses:**")
                for bullet in evidence["explanation"]["weaknesses"]:
                    st.markdown(f"- {bullet}")
            if evidence["explanation"]["strengths"]:
                st.markdown("**Strengths:**")
                for bullet in evidence["explanation"]["strengths"]:
                    st.markdown(f"- {bullet}")

    # ---- Grounding ----
    with col2:
        st.markdown(f"""
        <div class="confidence-metric">
            <div class="metric-label">Grounding quality</div>
            <div class="metric-value">{grounding['score']:.2f}</div>
            <div class="confidence-level"
                 style="color:{level_colors[grounding['level']]}">
                 {grounding['level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Why this score?"):
            if grounding["explanation"]["weaknesses"]:
                st.markdown("**Weaknesses:**")
                for bullet in grounding["explanation"]["weaknesses"]:
                    st.markdown(f"- {bullet}")
            if grounding["explanation"]["strengths"]:
                st.markdown("**Strengths:**")
                for bullet in grounding["explanation"]["strengths"]:
                    st.markdown(f"- {bullet}")


### Citations

def render_sentence_with_inline_citations(item, citation_style: CitationStyle):
    text = item["text"]
    citations = item.get("citations", [])

    if not citations:
        return f"- {text}"

    if citation_style == CitationStyle.NUMERIC:
        citation_str = ", ".join(citations)
        return f"- {text} [{citation_str}]"

    elif citation_style == CitationStyle.AUTHOR_YEAR:
        citation_str = "; ".join(citations)
        return f"- {text} [{citation_str}]"

    else:
        return f"- {text}"
    

### Limitations

def show_limitations(data, level=None):
    limitations = data.get("limitations", [])
    for lim in limitations:
        if level == "error":
            st.error(lim)
        elif level == "warning":
            st.warning(lim)
        else:
            st.write(f"{lim}")


### Metadata

def show_metadata(data):
    if data.get("meta", {}):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Metadata", expanded=False):
            st.json(data.get("meta", {}))


### Sources

def show_sources(data, citation_style=CitationStyle.NUMERIC):
    sources = data.get("sources")
    
    for src in sources:
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

### Grounding metrics

def show_grounding_metrics(data):
    metrics = data.get("grounding_metrics")

    if metrics is not None:
        with st.expander("Grounding Metrics", expanded=False):
            
            if not metrics:
                st.info("Grounding metrics are unavailable for this response.")
            
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Available chunks", metrics.get('available_chunks', 0))
                    st.metric("Used chunks", metrics.get('used_chunks', 0))
                    st.metric(
                        "Chunk coverage",
                        f"{metrics.get('chunk_coverage', 0):.0%}"
                    )

                with col2:
                    st.metric("Available papers", metrics.get('available_papers', 0))
                    st.metric("Used papers", metrics.get('used_papers', 0))
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

### Trace

def show_trace(data):
    trace = data.get("trace")
    
    with st.sidebar:
        show_trace = st.checkbox("Show diagnostics", value=False)
    
    if trace and show_trace:
        st.markdown("---")
        ## Query expansion
        query_expansion = trace.get("query_expansion", None)
        
        if query_expansion:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Query expansion")
            st.markdown(query_expansion)
        
        ## Strong hit chunks
        strong_hits = trace.get("strong_hit_chunks", None)
        
        if strong_hits:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Strong hit chunks: {len(strong_hits)}")
            for chunk in strong_hits:
                with st.expander(f"From: **{chunk['title']}** ({chunk['year']})", expanded=False):
                    st.markdown(f"{chunk['text']}")

        ## Evidence usage
        chunks = trace.get("chunks_provided_to_synthesizer", [])
        papers = trace.get("paper_stats", [])   
        
        if papers and chunks:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Evidence usage")

            for paper in papers:
                paper_id = paper["paper_id"]
                paper_chunks = [
                    c for c in chunks
                    if c["paper_id"] == paper_id
                ]

                # Paper header
                st.markdown(f"**📄 {paper['title']} ({paper['year']})**")
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
