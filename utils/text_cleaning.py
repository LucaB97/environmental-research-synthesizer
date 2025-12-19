import re
from collections import Counter


REFERENCE_HEADERS = [
    r"^references\s*$",
    r"^bibliography\s*$",
    r"^works cited\s*$",
]

def remove_references(text):
    """
    Remove the references or bibliography section from a document.

    The function looks for common section headers (e.g. 'References',
    'Bibliography') and truncates the text at the first occurrence.

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    str
        Text truncated before the references section, if found.
    """
    
    lines = text.split("\n")
    for i, line in enumerate(lines):
        for pattern in REFERENCE_HEADERS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                return "\n".join(lines[:i])
    return text


def clean_text(text):
    """
    Perform light whitespace normalization on extracted text.

    - Collapses multiple newlines into paragraph breaks
    - Converts single newlines into spaces
    - Normalizes repeated whitespace

    Parameters
    ----------
    text : str
        Raw extracted document text.

    Returns
    -------
    str
        Cleaned text with normalized whitespace.
    """

    # multiple newlines replaced with a paragraph break
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # single newlines replaced with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # spaces normalization
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_headers_footers(pages_lines, threshold=0.6, n_lines=2):
    """
    Remove recurring headers and footers across document pages.

    Lines appearing in the top or bottom `n_lines` of more than
    `threshold` proportion of pages are considered headers or footers
    and removed from all pages.

    Parameters
    ----------
    pages_lines : list[list[str]]
        List of pages, where each page is a list of text lines.
    threshold : float, optional
        Proportion of pages a line must appear in to be removed.
    n_lines : int, optional
        Number of top and bottom lines per page to consider.

    Returns
    -------
    list[list[str]]
        Pages with common headers and footers removed.
    """
    
    header_candidates = []
    footer_candidates = []

    for lines in pages_lines:
        if len(lines) >= n_lines:
            header_candidates.extend(lines[:n_lines])
            footer_candidates.extend(lines[-n_lines:])

    num_pages = len(pages_lines)

    header_counts = Counter(header_candidates)
    footer_counts = Counter(footer_candidates)

    common_headers = {
        line for line, count in header_counts.items()
        if count > threshold * num_pages
    }

    common_footers = {
        line for line, count in footer_counts.items()
        if count > threshold * num_pages
    }

    cleaned_pages = []
    for lines in pages_lines:
        cleaned = [
            line for line in lines
            if line not in common_headers
            and line not in common_footers
        ]
        cleaned_pages.append(cleaned)

    return cleaned_pages