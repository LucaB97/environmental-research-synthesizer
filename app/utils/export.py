import json
from typing import Dict, Any
import streamlit as st


def response_to_json(
    response: Dict,
    include_trace: bool = False
) -> str:
    """
    Serialize full query response to JSON.
    """

    payload = dict(response)

    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    else:
        payload = dict(response)

    if not include_trace:
        payload.pop("trace", None)

    return json.dumps(payload, indent=2)



def response_to_markdown(response: Dict[str, Any]) -> str:
    """
    Serialize query response to a human-readable Markdown report.
    """
    lines = []

    # Question
    lines.append("# Question\n")
    lines.append(response.get("question", []))

    # Synthesis
    lines.append("\n# Synthesized Answer\n")

    for item in response.get("answer", []):
        text = item["text"]
        citations = item.get("citations", [])

        if citations:
            text += " " + " ".join(f"[{c}]" for c in citations)

        lines.append(f"- {text}")

    # Limitations
    if response.get("limitations"):
        lines.append("\n# Limitations\n")
        for lim in response["limitations"]:
            lines.append(f"- {lim}")

    # Sources
    if response.get("sources"):
        lines.append("\n# Sources\n")
        for src in response["sources"]:
            author_year = f"{src['authors']} ({src['year']})"
            journal = f", {src['journal']}" if src.get("journal") else ""
            lines.append(
                f"- **{src['title']}** — {author_year}{journal}"
            )

    # # Metadata
    # meta = response.get("meta")
    # if meta:
    #     lines.append("\n# Metadata\n")
    #     for k, v in meta.items():
    #         lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")

    # # Grounding metrics
    # metrics = response.get("grounding_metrics")
    # if metrics:
    #     lines.append("\n## Grounding metrics\n")
    #     for k, v in metrics.items():
    #         lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")

    # Evidence metrics
    confidence = response.get("confidence")
    if confidence:
        lines.append("\n## Confidence profile\n")
        
        for axis_name, axis_data in confidence.items():
            lines.append(f"### {axis_name.capitalize()}")
            
            if isinstance(axis_data, dict):
                for k, v in axis_data.items():
                    lines.append(f"- **{k.capitalize()}**: {v}")
            else:
                lines.append(f"- {axis_data}")

    return "\n".join(lines)



def export_output(data):
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
                has_trace = bool(data.get("trace"))
                include_trace = st.checkbox(
                    "Include diagnostic info (chunks usage)",
                    value=False,
                    disabled=not has_trace
                )
                data_export = response_to_json(data, include_trace)
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