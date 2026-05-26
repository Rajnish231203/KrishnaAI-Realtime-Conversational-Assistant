"""
RAG Retriever — Krishna Voice Assistant
========================================
Performs semantic search over Bhagavad Gita verses using a FAISS index and
a SentenceTransformer embedding model.

Architecture:
    User query
        → SentenceTransformer embedding  (all-MiniLM-L6-v2)
        → L2-normalised vector
        → FAISS inner-product search     (equivalent to cosine similarity)
        → threshold filter
        → ranked List[VerseResult]

Similarity scores:
    Scores are cosine similarities in the range [0.0, 1.0].
    A score of 1.0 means the query and verse embeddings are identical.
    Config.RAG_SIMILARITY_THRESHOLD (default 0.35) is the minimum score
    required for a verse to be included in the results.

Singleton pattern:
    GitaRetriever uses __new__ to guarantee a single shared instance across
    the entire process, so the heavy embedding model and FAISS index are
    loaded only once regardless of how many callers instantiate the class.

Public interface (consumed by websocket/server.py):
    GitaRetriever().retrieve(query, top_k, threshold) → List[VerseResult]
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.app.config.config import Config   # fixed: was `from app.config.config`

# ---------------------------------------------------------------------------
# Path constants — single source of truth for all RAG store locations.
# RAG_DIR  = the directory containing this file  (.../services/rag/)
# RAG_STORE_DIR = rag_store/ subfolder inside it (.../services/rag/rag_store/)
# ---------------------------------------------------------------------------
RAG_DIR       = Path(__file__).resolve().parent
RAG_STORE_DIR = RAG_DIR / "rag_store"

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.info("RAG store dir: %s", RAG_STORE_DIR)


# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class VerseResult:
    """
    Structured container for a single retrieved Bhagavad Gita verse.

    Attributes:
        chapter:         Chapter number in the Gita (1–18).
        verse:           Verse number within the chapter.
        sanskrit:        Original Sanskrit text of the verse.
        translation:     English translation or combined meaning text.
        relevance_score: Cosine similarity to the user query (0.0–1.0).
        meaning:         Extended meaning or commentary (may be empty).
    """
    chapter:         int
    verse:           int
    sanskrit:        str
    translation:     str
    relevance_score: float
    meaning:         str = ""


# ===========================================================================
# Retriever
# ===========================================================================

class GitaRetriever:
    """
    Singleton semantic retriever for Bhagavad Gita verses.

    Lazy-loads the SentenceTransformer embedding model, FAISS index, and
    verse metadata on the first call to ``retrieve()``.  Subsequent calls
    reuse the already-loaded resources, keeping inference latency low.

    Usage::

        retriever = GitaRetriever()
        verses = retriever.retrieve("What is my duty in life?", top_k=5)
        for v in verses:
            print(v.chapter, v.verse, v.relevance_score)
    """

    # ------------------------------------------------------------------
    # Singleton bookkeeping
    # ------------------------------------------------------------------
    _instance: Optional["GitaRetriever"] = None

    def __new__(cls) -> "GitaRetriever":
        if cls._instance is None:
            logger.info("Initializing GitaRetriever singleton.")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation (runs once per process)
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        if self._initialized:
            return

        # Path constants are resolved at module level (RAG_STORE_DIR).
        # NOTE: faiss.read_index() requires a plain str, not a Path object.
        # Store as Path for existence checks; convert to str at the FAISS call.
        self.index_path: Path = RAG_STORE_DIR / "gita.faiss"
        self.meta_path:  Path = RAG_STORE_DIR / "gita_meta.json"
        self.model_name: str  = "all-MiniLM-L6-v2"

        # Lazily populated on first retrieve() call.
        self.model:       Optional[SentenceTransformer] = None
        self.index:       Optional[faiss.Index]         = None
        self.metadata:    List[dict]                    = []
        self._load_error: Optional[str]                 = None

        self._initialized = True
        logger.debug(
            "GitaRetriever configured | index: %s | meta: %s",
            self.index_path,
            self.meta_path,
        )

    # ------------------------------------------------------------------
    # Lazy resource loader
    # ------------------------------------------------------------------

    def _load_resources(self) -> bool:
        """
        Lazily load the embedding model, verse metadata, and FAISS index.

        Called automatically by ``retrieve()`` on the first invocation.
        Subsequent calls return immediately if resources are already loaded.

        Returns:
            True  — all resources loaded successfully.
            False — at least one resource failed to load; error is logged.
        """
        # Already loaded — fast path.
        # Bug fix: check metadata too to prevent partial-load edge cases
        # where model + index loaded but metadata deserialization failed.
        if self.model and self.index and self.metadata:
            return True

        logger.info("RAG: Loading resources (lazy load)...")
        start_time = time.time()

        try:
            # 1. Embedding model -------------------------------------------
            if not self.model:
                logger.info(
                    "Loading RAG embedding model: %s", self.model_name
                )
                self.model = SentenceTransformer(self.model_name)
                logger.debug("Embedding model loaded.")

            # 2. Verse metadata --------------------------------------------
            if not self.metadata:
                logger.info("Loading verse metadata from: %s", self.meta_path)
                if not self.meta_path.exists():
                    raise FileNotFoundError(
                        f"Verse metadata not found at '{self.meta_path}'. "
                        "Run index_builder.py to generate it."
                    )
                with open(self.meta_path, "r", encoding="utf-8") as fh:
                    self.metadata = json.load(fh)
                logger.debug(
                    "Verse metadata loaded: %d entries.", len(self.metadata)
                )

            # 3. FAISS index -----------------------------------------------
            if not self.index:
                logger.info("Loading FAISS index from: %s", self.index_path)
                if not self.index_path.exists():
                    raise FileNotFoundError(
                        f"FAISS index not found at '{self.index_path}'. "
                        "Run index_builder.py to generate it."
                    )
                # Bug fix: faiss.read_index() C++ binding requires a plain str,
                # not a pathlib.Path object.  The 'Wrong number or type of
                # arguments for overloaded function read_index' error is caused
                # by passing a Path here.
                self.index = faiss.read_index(str(self.index_path))
                logger.debug(
                    "FAISS index loaded: %d vectors.", self.index.ntotal
                )

            elapsed = time.time() - start_time
            logger.info("RAG: All resources loaded in %.2fs.", elapsed)
            return True

        except Exception as exc:
            logger.error("RAG resource load error: %s", exc)
            self._load_error = str(exc)
            return False

    # ------------------------------------------------------------------
    # Public retrieval method
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:     str,
        top_k:     Optional[int]   = None,
        threshold: Optional[float] = None,
    ) -> List[VerseResult]:
        """
        Retrieve Bhagavad Gita verses semantically similar to *query*.

        Steps:
            1. Embed the query using the SentenceTransformer model.
            2. L2-normalise the embedding so inner-product == cosine similarity.
            3. Run a FAISS k-nearest-neighbour search.
            4. Filter results below *threshold* (cosine similarity 0.0–1.0).
            5. Return ranked ``VerseResult`` objects.

        Args:
            query:     Natural-language question or statement from the user.
            top_k:     Maximum number of verses to return.
                       Defaults to ``Config.RAG_TOP_K`` (env: RAG_TOP_K).
            threshold: Minimum cosine-similarity score to include a verse.
                       Defaults to ``Config.RAG_SIMILARITY_THRESHOLD``
                       (env: RAG_SIMILARITY_THRESHOLD).

        Returns:
            List of ``VerseResult`` objects sorted by descending relevance.
            Returns an empty list if resources failed to load or an error
            occurred during retrieval.
        """
        # Apply Config-driven defaults when callers do not supply overrides.
        effective_top_k:    int   = top_k     if top_k     is not None else Config.RAG_TOP_K
        effective_threshold: float = threshold if threshold is not None else Config.RAG_SIMILARITY_THRESHOLD

        # Ensure model, index and metadata are available.
        if not self._load_resources():
            logger.error(
                "RAG retrieve() aborted — resources unavailable. "
                "Last load error: %s",
                self._load_error,
            )
            return []

        start_time = time.time()

        try:
            # ----------------------------------------------------------
            # 1. Embed and normalise query
            # ----------------------------------------------------------
            embedding: np.ndarray = self.model.encode(
                [query], convert_to_numpy=True
            )
            faiss.normalize_L2(embedding)

            # ----------------------------------------------------------
            # 2. FAISS nearest-neighbour search
            #    scores → inner-product (== cosine similarity after L2 norm)
            #    indices → position in metadata list
            # ----------------------------------------------------------
            scores, indices = self.index.search(embedding, effective_top_k)

            # ----------------------------------------------------------
            # 3. Build result list
            # ----------------------------------------------------------
            results: List[VerseResult] = []

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    # FAISS returns -1 for unfilled slots when the index
                    # contains fewer vectors than top_k.
                    logger.debug("Skipping FAISS slot idx=-1.")
                    continue

                if score < effective_threshold:
                    logger.debug(
                        "Verse idx=%d score=%.4f below threshold %.4f — skipped.",
                        idx, score, effective_threshold,
                    )
                    continue

                meta = self.metadata[idx]

                # Bug fix: index_builder.py stores word_meanings under the key
                # 'translation' (no separate 'meaning' key is written).
                # Prefer 'translation'; fall back to 'meaning' for legacy stores.
                word_meanings_text: str = (
                    meta.get("translation")
                    or meta.get("meaning")
                    or ""
                )

                result = VerseResult(
                    chapter=meta["chapter_number"],
                    verse=meta["verse_number"],
                    sanskrit=meta["sanskrit"],
                    translation=word_meanings_text,
                    relevance_score=float(score),
                    # Populate meaning with the same content so prompt_builder
                    # _format_verses() (which prefers .meaning) always has text.
                    meaning=word_meanings_text,
                )
                results.append(result)

                logger.debug(
                    "Matched verse Ch%d:%d | score=%.4f | preview: %s",
                    result.chapter,
                    result.verse,
                    result.relevance_score,
                    word_meanings_text[:60],
                )

            # ----------------------------------------------------------
            # 4. Latency metric
            # ----------------------------------------------------------
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "RAG retrieval completed in %sms with %d results "
                "(top_k=%d, threshold=%.2f).",
                f"{latency_ms:.0f}",
                len(results),
                effective_top_k,
                effective_threshold,
            )

            return results

        except Exception as exc:
            logger.error("RAG retrieval error: %s", exc)
            return []


# ===========================================================================
# CLI smoke-test
# ===========================================================================

if __name__ == "__main__":
    from app.config.config import setup_logging  # noqa: F401 (available if needed)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("Testing GitaRetriever...")
    retriever = GitaRetriever()
    verses    = retriever.retrieve("I am confused about my duty in life", top_k=3)

    for v in verses:
        print(
            f"Ch{v.chapter}:{v.verse} "
            f"(score={v.relevance_score:.2f}) — "
            f"{v.translation[:60]}..."
        )