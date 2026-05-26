"""
Builds the FAISS vector index for Bhagavad Gita verses.

This script is intended to be run during setup or when the dataset changes.

Output:
    rag_store/gita.faiss
    rag_store/gita_meta.json
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# RAG_DIR       = the directory containing this file  (.../services/rag/)
# RAG_STORE_DIR = rag_store/ subfolder inside it      (.../services/rag/rag_store/)
# DATASET_PATH  = geeta_data/verse.json at project root (3 parents up from rag/)
# ---------------------------------------------------------------------

RAG_DIR       = Path(__file__).resolve().parent
RAG_STORE_DIR = RAG_DIR / "rag_store"

# parents[0]=services  parents[1]=app  parents[2]=backend  parents[3]=voice_ass
DATASET_PATH  = RAG_DIR.parents[4] / "geeta_data" / "verse.json"

INDEX_PATH    = RAG_STORE_DIR / "gita.faiss"
META_PATH     = RAG_STORE_DIR / "gita_meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"

logger.info("RAG store dir: %s", RAG_STORE_DIR)

# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------


def load_verses() -> List[dict]:
    """
    Load verses from dataset.

    Returns
    -------
    List[dict]
        Raw verse objects from JSON.
    """

    logger.info("Loading dataset from %s", DATASET_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        verses = json.load(f)

    logger.info("Loaded %d verses", len(verses))

    return verses


# ---------------------------------------------------------------------
# Embedding Generation
# ---------------------------------------------------------------------


def create_embeddings(verses: List[dict]) -> Tuple[np.ndarray, List[str]]:
    """
    Generate sentence embeddings for all verses.

    Parameters
    ----------
    verses : List[dict]

    Returns
    -------
    embeddings : np.ndarray
    texts : List[str]
    """

    logger.info("Loading embedding model: %s", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    texts: List[str] = []

    logger.info("Preparing verse chunks")

    for verse in verses:

        chapter = verse.get("chapter_number")
        verse_no = verse.get("verse_number")

        sanskrit = verse.get("text", "").strip()
        meaning = verse.get("word_meanings", "").strip()

        text_chunk = (
            f"Chapter {chapter} Verse {verse_no}\n"
            f"Sanskrit: {sanskrit}\n"
            f"Meaning: {meaning}"
        )

        texts.append(text_chunk)

    logger.info("Generating embeddings for %d verses", len(texts))

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    logger.info("Embedding generation completed")

    return embeddings, texts


# ---------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS index.

    Parameters
    ----------
    embeddings : np.ndarray

    Returns
    -------
    faiss.Index
    """

    logger.info("Building FAISS index")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    logger.info("FAISS index built with %d vectors", index.ntotal)

    return index


# ---------------------------------------------------------------------
# Save Artifacts
# ---------------------------------------------------------------------


def save_artifacts(index: faiss.IndexFlatIP, verses: List[dict]) -> None:
    """
    Save FAISS index and metadata.
    """

    logger.info("Saving artifacts to %s", RAG_STORE_DIR)

    RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))

    logger.info("Saved FAISS index → %s", INDEX_PATH)

    # Save metadata

    metadata = []

    for i, verse in enumerate(verses):

        meta = {
            "id": i,
            "chapter_number": verse.get("chapter_number"),
            "verse_number": verse.get("verse_number"),
            "sanskrit": verse.get("text", "").strip(),
            "transliteration": verse.get("transliteration", "").strip(),
            "translation": verse.get("word_meanings", "").strip(),
        }

        metadata.append(meta)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Saved metadata → %s", META_PATH)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    start_time = time.time()

    logger.info("Starting Bhagavad Gita RAG index builder")

    try:

        verses = load_verses()

        embeddings, texts = create_embeddings(verses)

        index = build_index(embeddings)

        save_artifacts(index, verses)

        duration = time.time() - start_time

        logger.info("Index built successfully in %.2f seconds", duration)

    except Exception:
        logger.exception("Index building failed")
        raise


if __name__ == "__main__":
    main()