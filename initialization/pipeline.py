from pathlib import Path
from .config import InitializationConfig
from .extraction import extract_chunks
from .indexing import build_index_pipeline
from .tuning import run_tuning


def initialize_system(config: InitializationConfig):
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = PROJECT_ROOT / "data"

    papers_dir = data_dir / "papers"
    metadata_path = data_dir / "metadata.csv"

    if not papers_dir.exists() or not metadata_path.exists():
        raise ValueError("Initialization failed: Missing data")

    chunks_path = data_dir / f"chunks_{config.chunk_size}t_{config.overlap}o.json"
    index_path = data_dir / f"faiss_{config.embedding}_{config.chunk_size}t_{config.overlap}o.index"
    params_path = data_dir / f"parameters_{config.embedding}_{config.chunk_size}t_{config.overlap}o.json"

    # --- Step 1: extraction ---
    if not chunks_path.exists():
        if not config.auto_build:
            raise RuntimeError("Missing chunks and auto_build is disabled")
        print("Running TEXT EXTRACTION & CHUNKING")
        extract_chunks(
            config,
            papers_dir=papers_dir,
            metadata_path=metadata_path,
            output_path=chunks_path
        )

    # --- Step 2: indexing ---
    if not index_path.exists():
        if not config.auto_build:
            raise RuntimeError("Missing index and auto_build is disabled")
        print("Running INDEXING")
        build_index_pipeline(config, chunks_path, index_path)

    # --- Step 3: tuning ---
    if not params_path.exists():
        if not config.auto_build:
            raise RuntimeError("Missing parameters and auto_build is disabled")
        print("Running PARAMETERS TUNING")
        run_tuning(config, chunks_path, index_path, params_path)

    return {
        "metadata_path": metadata_path,
        "chunks_path": chunks_path,
        "index_path": index_path,
        "params_path": params_path
    }