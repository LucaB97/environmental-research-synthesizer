import pdfplumber

def extract_text_two_columns(page):
    """
    Extract text from a PDF page assuming a two-column layout.

    The page is split vertically into left and right halves, which are
    extracted separately and then concatenated in reading order.

    Parameters
    ----------
    page : pdfplumber.page.Page
        A pdfplumber Page object.

    Returns
    -------
    str
        Extracted text with left column followed by right column.
    """   
     
    width = page.width
    midpoint = width / 2

    left_bbox = (0, 0, midpoint, page.height)
    right_bbox = (midpoint, 0, width, page.height)

    left = page.crop(left_bbox).extract_text(layout=True) or ""
    right = page.crop(right_bbox).extract_text(layout=True) or ""

    return left + "\n\n" + right