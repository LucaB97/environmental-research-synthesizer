def create_chunks(text, metadata, idx, chunk_size=400, overlap=50):
    """
    Split document text into overlapping character-level chunks
    and attach paper metadata to each chunk.

    Parameters
    ----------
    text : str
        Cleaned document text.
    metadata : pandas.DataFrame
        Metadata table containing paper information.
    idx : int
        Row index of the current paper in the metadata DataFrame.
    chunk_size : int, optional
        Number of characters per chunk.
    overlap : int, optional
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[dict]
        List of chunk dictionaries ready for JSON serialization.
    """
    
    data = []
    n_chunks = (len(text) + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        
        start = i * (chunk_size - overlap)
        end = start + chunk_size
        if end > len(text):
            end = len(text)

        chunk_text = text[start:end]
        chunk_id =  metadata.paper_id.iloc[idx] + "__chunk_" + str(i)
        new_entry = {
            "chunk_id": chunk_id,
            "paper_id": str(metadata.paper_id.iloc[idx]),
            "title": str(metadata.title.iloc[idx]),
            "year": int(metadata.year.iloc[idx]),
            "text": chunk_text
        }
        data.append(new_entry)

    return data