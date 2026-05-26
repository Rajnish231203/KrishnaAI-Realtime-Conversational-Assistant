"""
RAG Prompt Builder — Krishna Voice Assistant
=============================================
Constructs LLM prompts that ground Krishna's responses in specific
Bhagavad Gita verses retrieved by ``GitaRetriever``.

Design philosophy:
    The prompt is the bridge between semantic retrieval and language
    generation.  By injecting retrieved verses as the *only* permitted
    knowledge source we prevent hallucination of fictitious citations
    while keeping responses anchored in authentic scriptural content.

Verse formatting:
    Each ``VerseResult`` is rendered as a labelled block::

        [VERSE 1]
        Reference: Bhagavad Gita 2.47
        Sanskrit:  ...
        Meaning:   ...

    This structured layout helps the LLM clearly demarcate source
    material from its own generated text and makes citation natural.

Public interface (consumed by websocket/server.py and streaming_llm.py):
    build_krishna_prompt(user_query, retrieved_verses) → str
"""

import logging
from typing import List

from backend.app.services.rag.retriever import VerseResult

# ---------------------------------------------------------------------------
# Module-level logger — activated by setup_logging() in the server entrypoint.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# Prompt builder
# ===========================================================================

def build_krishna_prompt(
    user_query: str,
    retrieved_verses: List[VerseResult],
    conversation_history: list | None = None,
    response_mode: str = "general",
) -> str:
    """
    Construct a grounded RAG prompt for the Krishna LLM persona.

    The prompt:
        1. Establishes Krishna's identity and tone.
        2. Injects the user's question as the ``CONTEXT`` block.
        3. Provides all retrieved verses as the exclusive ``KNOWLEDGE BASE``.
        4. Issues strict instructions that forbid hallucinated citations.
        5. Specifies the desired response language and format.

    Handling empty verse lists:
        If no verses were retrieved a compassionate fallback prompt is
        returned that instructs the LLM to respond from general Gita
        wisdom without fabricating specific verse references.

    Args:
        user_query:       Natural-language question from the user.
        retrieved_verses: Verses returned by ``GitaRetriever.retrieve()``.
                          May be empty if retrieval found no matches above
                          the similarity threshold.

    Returns:
        Fully-formed prompt string ready to be passed to the LLM.
    """
    logger.info(
        "Building Krishna prompt | verses: %d | query: %s…",
        len(retrieved_verses),
        user_query[:60],
    )

    # ------------------------------------------------------------------
    # Guard: no verses retrieved
    # ------------------------------------------------------------------
    if not retrieved_verses:
        logger.warning(
            "No verses retrieved for query: '%s'. "
            "Returning compassionate fallback prompt.",
            user_query,
        )
        return _build_fallback_prompt(user_query)

    # ------------------------------------------------------------------
    # Unified language instruction (Hinglish)
    # ------------------------------------------------------------------
    if response_mode == "scripture":
        lang_instruction = (
            "Respond in natural conversational Hinglish using a smooth mix "
            "of Hindi and English naturally. Include exact Bhagavad Gita "
            "chapter references, verse references, and Sanskrit shlokas "
            "when genuinely relevant to the user's request. "
            "Explain the spiritual meaning clearly and conversationally."
        )
    else:
        lang_instruction = (
            "Respond in natural conversational Hinglish using a smooth mix "
            "of Hindi and English naturally. Keep the language simple, calm, "
            "human, emotionally intelligent, and conversational. "
            "Avoid explicit chapter and verse references unless the user "
            "directly asks for them. Avoid sounding academic or robotic."
        )

    # ------------------------------------------------------------------
    # Format retrieved verses as labelled blocks
    # ------------------------------------------------------------------
    verses_text = _format_verses(
        retrieved_verses,
        response_mode=response_mode,
    )

    # Log each verse reference used (visible at DEBUG level only).
    for v in retrieved_verses:
        logger.debug(
            "Using verse: Ch%d.%d | score=%.4f | preview: %s…",
            v.chapter,
            v.verse,
            v.relevance_score,
            v.meaning[:50] if v.meaning else v.translation[:50],
        )

    # ------------------------------------------------------------------
    # Conversation history context (last 4 turns = 2 exchanges)
    # ------------------------------------------------------------------
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]  # keep prompt size bounded
        history_lines = []
        for msg in recent:
            role = "Seeker" if msg["role"] == "user" else "Krishna"
            history_lines.append(f"{role}: {msg['content'][:120]}")
        if history_lines:
            history_text = (
                "\nCONVERSATION HISTORY (for context only — do NOT repeat these):\n"
                + "\n".join(history_lines)
                + "\n"
            )

    # ------------------------------------------------------------------
    # Assemble the final prompt
    # ------------------------------------------------------------------

    if response_mode == "scripture":
        citation_instruction = (
            "Mention exact chapter references, verse references, and Sanskrit "
            "shlokas naturally when helpful. The user explicitly wants "
            "scriptural grounding."
        )
    else:
        citation_instruction = (
            "Avoid explicit chapter and verse references unless absolutely "
            "necessary. Speak naturally and conversationally, as if Krishna "
            "is guiding a friend in everyday life. You may indirectly refer "
            "to Gita wisdom naturally without sounding academic."
        )

    prompt = f"""You are a wise Krishna-inspired guide rooted in the teachings of the Bhagavad Gita.

CONTEXT — User Query:
"{user_query}"
{history_text}
KNOWLEDGE BASE (use the retrieved verses as the primary grounding for your response — do NOT invent or paraphrase verses not listed here):
{verses_text}

INSTRUCTIONS:
1. ANSWER the user's question directly first in a natural conversational way.
2. Use the retrieved teachings as grounding for your wisdom and guidance.
3. Explain the practical meaning naturally using relatable modern language.
4. {citation_instruction}
5. Avoid sounding academic, robotic, preachy, or overly philosophical.
6. {lang_instruction}
7. TONE: Calm, grounded, compassionate, and practical. Avoid theatrics.
8. ADDRESS the user warmly and occasionally, but avoid repeating the same phrases.
9. AVOID repetitive spiritual metaphors or cliches (rivers, lotus flowers, destiny, cosmic energy).
10. Keep responses concise, natural, and conversational; avoid being overly philosophical or preachy.
11. Speak in a way that feels understandable and relatable to ordinary modern conversation.
12. Do not assume the user is spiritually knowledgeable.
13. FORMAT:
   - Speak as if in direct conversation — avoid bullet points or numbered lists.
    - Limit your response to 3–5 sentences of clear and meaningful guidance.
   - Do NOT say "I am an AI", "based on the documents", or any similar disclaimer.
    - Do not sound like a motivational quote generator.

GOAL:
Help the user with grounded, practical, compassionate guidance using the specific truths \
contained in the retrieved verses above. Do not introduce teachings, \
stories, or citations from outside the provided KNOWLEDGE BASE.
"""

    logger.debug(
        "Prompt construction complete | total length: %d chars.", len(prompt)
    )
    return prompt


# ===========================================================================
# Private helpers
# ===========================================================================

def _format_verses(
    verses: List[VerseResult],
    response_mode: str = "general",
) -> str:
    """
    Render a list of ``VerseResult`` objects as a structured text block.

    Each verse is rendered as::

        [VERSE 1]
        Reference: Bhagavad Gita <chapter>.<verse>
        Sanskrit:  <original Sanskrit text>
        Meaning:   <translation / meaning>

    Blank lines between verses improve visual separation and help LLMs
    recognise where one source ends and the next begins.

    Args:
        verses: Non-empty list of retrieved verse objects.

    Returns:
        Multi-line string containing all formatted verse blocks.
    """
    blocks: List[str] = []

    for i, verse in enumerate(verses, start=1):
        # Prefer the 'meaning' field; fall back to 'translation' if absent.
        display_text = verse.meaning if verse.meaning else verse.translation

        if response_mode == "scripture":
            block = (
                f"[VERSE {i}]\n"
                f"Reference: Bhagavad Gita {verse.chapter}.{verse.verse}\n"
                f"Sanskrit:  {verse.sanskrit}\n"
                f"Meaning:   {display_text}"
            )
        else:
            block = (
                f"[TEACHING {i}]\n"
                f"Core Teaching: {display_text}"
            )
        blocks.append(block)

    return "\n\n".join(blocks)


def _build_fallback_prompt(user_query: str) -> str:
    """
    Construct a compassionate fallback prompt when no verses were retrieved.

    Instructs the LLM to respond from the general spirit of Gita wisdom
    without citing specific verses, so the user still receives a meaningful
    and on-brand response rather than an error or silence.

    Args:
        user_query: Original user question.

    Returns:
        Fallback prompt string.
    """
    lang_instruction = (
        "Respond in natural conversational Hinglish using a smooth mix "
        "of Hindi and English naturally. Keep the language simple, calm, "
        "human, emotionally intelligent, and conversational."
    )

    return f"""You are a wise Krishna-inspired guide rooted in the teachings of the Bhagavad Gita.

CONTEXT — User Query:
"{user_query}"

NOTE: No specific verse references are available for this query.
Respond from the general spirit and universal wisdom of the Bhagavad Gita.
Do NOT fabricate or invent specific verse numbers or citations.

INSTRUCTIONS:
1. ANSWER the user's question directly first, then add Gita-based perspective.
2. Offer compassionate, practical guidance rooted in Gita philosophy.
3. {lang_instruction}
4. TONE: Calm, grounded, and encouraging. Avoid theatrics.
5. Use warm, conversational addressing occasionally without repeating the same phrases.
6. Avoid repetitive spiritual metaphors or cliches (rivers, lotus flowers, destiny, cosmic energy).
7. Keep responses concise, natural, and conversational; avoid being overly philosophical or preachy.
8. Speak in a way that feels understandable and relatable to ordinary modern conversation.
9. Do not assume the user is spiritually knowledgeable.
10. Limit your response to 2–4 sentences.
11. Do NOT say "I am an AI" or any similar disclaimer.
"""