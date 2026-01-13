def render_sentence_with_inline_citations(item):
    text = item["text"]
    citations = item.get("citations", [])

    if citations:
        citation_str = "; ".join(citations)
        return f"- {text} [{citation_str}]"
    else:
        return f"- {text}"