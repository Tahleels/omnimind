"""
youtube_parser.py — YouTube Transcript Extractor for Dynamic Ingestion (v2)

Uses youtube-transcript-api to fetch the auto-generated or manual
captions for a YouTube video and convert them into a clean text
transcript suitable for chunking and indexing.

Why transcripts instead of video download?
  - No ffmpeg/heavy dependencies required.
  - Text-only = much smaller storage footprint.
  - Works for any video with captions (auto-generated or manual).
  - Fast: typically < 500ms to fetch.

Supported URL formats:
  - https://www.youtube.com/watch?v=VIDEO_ID
  - https://youtu.be/VIDEO_ID
  - https://www.youtube.com/watch?v=VIDEO_ID&t=123s (with timestamp params)

Limitations:
  - Only works for videos with available captions.
  - Age-restricted or private videos are not supported.
  - Transcript language preference: English first, then any available.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

from src.v2.ingestion.parsers.url_parser import ParsedDocument

logger = logging.getLogger(__name__)

# Patterns to extract video ID from various YouTube URL formats
_VIDEO_ID_PATTERNS = [
    re.compile(r"youtube\.com/watch\?.*?v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
]

# Language preference order for transcript fetching
_LANGUAGE_PREFERENCE = ["en", "en-US", "en-GB"]


def _extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID from a URL."""
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def _fetch_transcript(video_id: str) -> tuple[list[dict], str]:
    """
    Fetch the transcript for a video ID.

    Returns:
        (transcript_entries, language_used)

    Raises:
        NoTranscriptFound if no captions are available.
    """
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try English first, then any manually created, then any auto-generated
    try:
        transcript = transcript_list.find_transcript(_LANGUAGE_PREFERENCE)
        return transcript.fetch(), transcript.language_code
    except Exception:
        pass

    # Fallback: take whichever transcript is available
    try:
        transcript = transcript_list.find_generated_transcript(_LANGUAGE_PREFERENCE)
        return transcript.fetch(), transcript.language_code
    except Exception:
        pass

    # Last resort: iterate and take first
    for transcript in transcript_list:
        return transcript.fetch(), transcript.language_code

    raise NoTranscriptFound(video_id, _LANGUAGE_PREFERENCE, {})


def _format_transcript(entries: list[dict]) -> str:
    """
    Convert transcript entries into clean paragraph-based text.

    Groups entries into ~5-minute blocks with timestamps to preserve
    navigation context, which helps the chunker split on natural breaks.

    Args:
        entries: List of {text, start, duration} dicts from the API.

    Returns:
        Formatted transcript string.
    """
    blocks: list[str] = []
    current_block: list[str] = []
    block_start: float = 0.0
    BLOCK_SECONDS = 300  # 5-minute blocks

    for entry in entries:
        start = entry.get("start", 0)
        text = entry.get("text", "").strip()
        if not text:
            continue

        if not current_block:
            block_start = start

        current_block.append(text)

        # Start a new block every BLOCK_SECONDS
        if start - block_start >= BLOCK_SECONDS:
            minutes = int(block_start // 60)
            seconds = int(block_start % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            blocks.append(f"{timestamp}\n{' '.join(current_block)}")
            current_block = []
            block_start = start

    # Flush remaining entries
    if current_block:
        minutes = int(block_start // 60)
        seconds = int(block_start % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        blocks.append(f"{timestamp}\n{' '.join(current_block)}")

    return "\n\n".join(blocks)


def parse_youtube(url: str) -> ParsedDocument:
    """
    Fetch and format the transcript of a YouTube video.

    Args:
        url: Full YouTube URL (watch or shortened youtu.be format).

    Returns:
        ParsedDocument with the video transcript as text.

    Raises:
        ValueError: If the URL is not a valid YouTube URL.
        RuntimeError: If no transcript is available for the video.
    """
    video_id = _extract_video_id(url)
    if video_id is None:
        raise ValueError(
            f"Could not extract a YouTube video ID from URL: {url}\n"
            "Supported formats: youtube.com/watch?v=ID or youtu.be/ID"
        )

    logger.info(f"Fetching YouTube transcript for video ID: {video_id}")

    try:
        entries, language = _fetch_transcript(video_id)
    except NoTranscriptFound as exc:
        raise RuntimeError(
            f"No captions available for video {video_id}.\n"
            "The video may have captions disabled or be age-restricted."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch transcript for {video_id}: {exc}"
        ) from exc

    transcript_text = _format_transcript(entries)

    if not transcript_text.strip():
        raise RuntimeError(
            f"Transcript for video {video_id} was empty after formatting."
        )

    # Use video ID as title stub — caller can override with the actual title
    title = f"YouTube Video: {video_id}"

    logger.info(
        f"✓ Parsed YouTube transcript: {len(transcript_text)} chars, "
        f"language={language}, video={video_id}"
    )

    return ParsedDocument(
        text=transcript_text,
        title=title,
        origin=url,
        source_type="youtube",
    )
