"""
citation_validator.py — Post-Processing Citation Validation

After the LLM generates an answer, this module:
1. Extracts all [n] citation references from the text.
2. Validates that each n ≤ number of context chunks provided.
3. Flags or strips any hallucinated citation numbers.

This is a critical production pattern — LLMs can generate [7] when only
5 chunks were provided, which would mislead users into believing a
nonexistent source was cited.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of citation validation."""
    original_text: str
    validated_text: str
    cited_indices: list[int]         # All [n] references found (1-indexed)
    invalid_citations: list[int]     # Citations that exceed num_chunks
    is_valid: bool                   # True if no hallucinated citations


def validate_citations(
    answer: str,
    num_chunks: int,
    strip_invalid: bool = True,
) -> ValidationResult:
    """
    Validate citation references in an LLM-generated answer.

    Args:
        answer: The raw answer text from the LLM.
        num_chunks: The number of context chunks that were provided.
        strip_invalid: If True, remove invalid [n] refs from the answer.
                       If False, replace with [INVALID_CITATION].

    Returns:
        ValidationResult with cleaned text and validation metadata.
    """
    # Extract all [n] style citations
    citation_pattern = re.compile(r'\[(\d+)\]')
    all_matches = citation_pattern.findall(answer)
    cited_indices = [int(m) for m in all_matches]

    # Find out-of-range citations (n must be between 1 and num_chunks)
    invalid_citations = [n for n in cited_indices if n < 1 or n > num_chunks]

    validated_text = answer
    if invalid_citations:
        invalid_set = {n for n in invalid_citations}
        logger.warning(
            f"⚠ Hallucinated citation(s) detected: "
            f"{sorted(invalid_set)} (only {num_chunks} chunks available). "
            f"{'Stripping' if strip_invalid else 'Flagging'} them."
        )

        if strip_invalid:
            # Remove hallucinated [n] references entirely
            def replace_invalid(match: re.Match) -> str:
                n = int(match.group(1))
                return "" if n in invalid_set else match.group(0)
            validated_text = citation_pattern.sub(replace_invalid, answer)
            # Clean up any double spaces left behind
            validated_text = re.sub(r'  +', ' ', validated_text).strip()
        else:
            # Replace with explicit marker
            def mark_invalid(match: re.Match) -> str:
                n = int(match.group(1))
                return f"[INVALID_CITATION:{n}]" if n in invalid_set else match.group(0)
            validated_text = citation_pattern.sub(mark_invalid, answer)

    return ValidationResult(
        original_text=answer,
        validated_text=validated_text,
        cited_indices=sorted(set(cited_indices)),
        invalid_citations=sorted(set(invalid_citations)),
        is_valid=len(invalid_citations) == 0,
    )


def extract_cited_sources(
    cited_indices: list[int],
    chunks: list,
) -> list[dict]:
    """
    Build a list of source references for the answer, based on which
    chunk indices were actually cited.

    Args:
        cited_indices: 1-indexed citation numbers found in the answer.
        chunks: The list of RetrievedChunk objects used as context.

    Returns:
        List of dicts with source info for the frontend to render.
    """
    sources = []
    seen_urls = set()
    for idx in cited_indices:
        chunk_idx = idx - 1  # Convert 1-indexed to 0-indexed
        if 0 <= chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            if chunk.source_url not in seen_urls:
                sources.append({
                    "citation_number": idx,
                    "title": chunk.source_title,
                    "url": chunk.source_url,
                    "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                })
                seen_urls.add(chunk.source_url)
    return sources
