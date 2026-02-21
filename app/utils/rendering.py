import matplotlib.pyplot as plt
import streamlit as st

from utils.citations import CitationStyle


###Confidence profile

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


###Citations

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
    

###Limitations

def show_limitations(data, level="warning"):
    limitations = data.get("limitations", [])
    for lim in limitations:
        if level == "error":
            st.error(lim)
        else:
            st.warning(lim)


###Metadata

def show_metadata(data):
    if data.get("meta", {}):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Metadata", expanded=False):
            st.json(data.get("meta", {}))