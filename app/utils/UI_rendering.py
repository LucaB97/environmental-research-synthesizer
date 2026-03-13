import streamlit as st
from utils.citations import CitationStyle


### Confidence profile

def format_explanation(expl):
    """
    Formats explanation for tooltip rendering.
    Supports:
      - None
      - string (semantic axis)
      - dict with 'strengths' and/or 'weaknesses'
    """

    if expl is None:
        return "No notable signals."

    # --- Case 1: simple string (semantic axis) ---
    if isinstance(expl, str):
        return f"<p style='margin-bottom:4px;'>{expl}</p><ul>"

    # --- Case 2: structured dictionary ---
    if isinstance(expl, dict):
        bullets = ""

        weaknesses = expl.get("weaknesses", [])
        strengths = expl.get("strengths", [])

        if weaknesses:
            bullets += "<p style='font-weight:600; margin-bottom:4px;'>Weaknesses</p><ul>"
            for w in weaknesses:
                bullets += f"<li>{w}</li>"
            bullets += "</ul><br>"

        if strengths:
            bullets += "<p style='font-weight:600; margin-bottom:4px;'>Strengths</p><ul>"
            for s in strengths:
                bullets += f"<li>{s}</li>"
            bullets += "</ul>"

        if bullets == "":
            return "No notable signals."

        return bullets

    # --- Fallback (unexpected type) ---
    return "No notable signals."


def render_confidence_profile(confidence):

    st.subheader("Confidence Profile")
    semantic = confidence["semantic"]
    evidence = confidence["evidence"]
    grounding = confidence["grounding"]

    level_colors = {
        "Strong": "#007acc",
        "Moderate": "#a17fcf",
        "Weak": "#888888",
        "Not_applicable": "#aaaaaa"
    }

    st.markdown("""
    <style>
    .confidence-metric {
        margin-bottom: 1.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
        font-weight: 600;
    }

    .metric-caption {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 0.5rem;
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

    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-left: 6px;
        color: #999;
        font-weight: 600;
        font-size: 0.8rem;   /* <-- add this */
        vertical-align: middle;  /* optional, improves alignment */
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 260px;
        background-color: #f9f9f9;
        color: #333;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        top: 125%;
        left: 50%;
        margin-left: -130px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 0.8rem;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
    }

    .tooltiptext ul {
        padding-left: 18px;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    semantic_caption = "How closely the top retrieved passages match the query"
    evidence_caption = "How much relevant evidence appears among the top passages"
    grounding_caption = "How well the answer integrates and balances the cited evidence"

    # ---- Semantic alignment ----
    with col1:
        semantic_tooltip = format_explanation(semantic["explanation"])

        st.markdown(f"""
        <div class="confidence-metric">
            <div class="metric-label">
                Semantic alignment
            </div>
            <div class="metric-caption">
                {semantic_caption}
            </div>
            <div class="metric-value">
                {semantic['score']:.2f}
                <span class="tooltip">ⓘ
                    <span class="tooltiptext">{semantic_tooltip}</span>
                </span>    
            </div>
            <div class="confidence-level"
                 style="color:{level_colors[semantic['level']]}">
                 {semantic['level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Evidence structure ----
    with col2:
        evidence_tooltip = format_explanation(evidence["explanation"])

        st.markdown(f"""
        <div class="confidence-metric">
            <div class="metric-label">
                Evidence structure
            </div>
            <div class="metric-caption">
                {evidence_caption}
            </div>
            <div class="metric-value">
                {evidence['score']:.2f}
                <span class="tooltip">ⓘ
                    <span class="tooltiptext">{evidence_tooltip}</span>
                </span>
            </div>
            <div class="confidence-level"
                 style="color:{level_colors[evidence['level']]}">
                 {evidence['level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Grounding quality ----
    with col3:
        grounding_tooltip = format_explanation(grounding["explanation"])

        st.markdown(f"""
        <div class="confidence-metric">
            <div class="metric-label">
                Grounding quality
            </div>
            <div class="metric-caption">
                {grounding_caption}                                    
            </div>
            <div class="metric-value">
                {grounding['score']:.2f}
                <span class="tooltip">ⓘ
                    <span class="tooltiptext">{grounding_tooltip}</span>
                </span>
            </div>
            <div class="confidence-level"
                 style="color:{level_colors[grounding['level']]}">
                 {grounding['level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if semantic['score'] < 0.5:
        st.warning("⚠️ Retrieved evidence is weakly aligned with the query")


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
            st.markdown(f"**Original query:**\n{query_expansion[0]}")
            st.markdown(f"**Expanded query:**\n{query_expansion[1]}")
        
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
