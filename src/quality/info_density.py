"""
Module 2a: Information Density Scorer

Measures how much useful information a video delivers per unit time.
A 5-min video that covers 10 key concepts is more information-dense
than a 20-min video covering the same 10 concepts with filler.

Scoring hierarchy (best to worst data source):
1. Transcript available → extract entities, claims, unique facts per minute
2. Title + description only → use factual signals and structure heuristics

When a transcript is available, we get a FAR more accurate score because
we're analyzing the actual spoken content, not just the metadata.
"""

import re
from typing import Optional


# ── Filler / low-info patterns in transcripts ──
FILLER_PATTERNS = [
    r"\b(um|uh|like|you know|basically|actually|literally|right)\b",
    r"\b(subscribe|notification|bell|like button|comment below|hit that)\b",
    r"\b(don't forget to|make sure to|link in the description|check out)\b",
    r"\b(sponsor|sponsored|promo code|discount|affiliate)\b",
]

# ── Factual claim patterns ──
FACTUAL_PATTERNS = [
    r'\b\d+\.?\d*\s*%',                           # Percentages: "45%"
    r'\b\d{4}\b',                                   # Years: "2024"
    r'\b\d+[\.,]?\d*\s*(million|billion|trillion)',  # Large numbers
    r'\b(study|research|according to|found that)\b', # Research citations
    r'\b(increase|decrease|growth|decline)\b',       # Trend indicators
    r'\b(published|journal|university|professor)\b', # Academic references
]


def _count_factual_signals(text: str) -> int:
    """Count indicators of factual content in text."""
    if not text:
        return 0
    signals = 0
    signals += len(re.findall(r'\d+\.?\d*%?', text))
    signals += len(re.findall(r'\b(19|20)\d{2}\b', text))
    signals += len(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b', text))
    return signals


def _has_structure(description: str) -> float:
    """Check for structural content indicators in the description."""
    if not description:
        return 0.0
    score = 0.0
    timestamps = len(re.findall(r'\d{1,2}:\d{2}', description))
    score += min(timestamps * 0.1, 0.4)
    numbered = len(re.findall(r'^\s*\d+[\.\)]\s', description, re.MULTILINE))
    score += min(numbered * 0.05, 0.3)
    bullets = len(re.findall(r'^\s*[-•✔✅▶]\s', description, re.MULTILINE))
    score += min(bullets * 0.05, 0.3)
    return min(score, 1.0)


def _compute_transcript_density(transcript: str, duration_minutes: float) -> float:
    """
    Compute information density from a transcript.
    This is far more accurate than description-only analysis because
    we're analyzing the actual spoken words.

    Metrics:
    - Unique factual claims per minute
    - Vocabulary richness (type-token ratio)
    - Filler word ratio (lower = more information-dense)
    - Factual pattern density
    """
    words = transcript.lower().split()
    total_words = len(words)

    if total_words < 10 or duration_minutes <= 0:
        return 0.0

    # 1. Vocabulary richness — Type-Token Ratio (unique words / total words)
    #    Higher TTR = more diverse vocabulary = more concepts covered
    unique_words = len(set(words))
    ttr = unique_words / total_words
    # TTR typically ranges 0.1–0.5 for speech; normalize to [0, 1]
    vocab_score = min(ttr / 0.4, 1.0)

    # 2. Filler word ratio — lower is better
    filler_count = sum(len(re.findall(p, transcript, re.IGNORECASE)) for p in FILLER_PATTERNS)
    filler_ratio = filler_count / total_words
    # Invert: low filler = high score
    filler_score = max(0.0, 1.0 - filler_ratio * 10)

    # 3. Factual pattern density — claims, stats, references per minute
    factual_hits = sum(len(re.findall(p, transcript, re.IGNORECASE)) for p in FACTUAL_PATTERNS)
    facts_per_minute = factual_hits / max(duration_minutes, 1)
    factual_score = min(facts_per_minute / 3.0, 1.0)

    # 4. Words per minute — too few (padded) or too many (rushed) is bad
    wpm = total_words / duration_minutes
    # Optimal range: 120–180 WPM for educational content
    if 120 <= wpm <= 180:
        pace_score = 1.0
    elif 100 <= wpm < 120 or 180 < wpm <= 220:
        pace_score = 0.7
    elif 80 <= wpm < 100 or 220 < wpm <= 260:
        pace_score = 0.4
    else:
        pace_score = 0.2

    # 5. Named entity density (rough: capitalized words in transcript)
    caps_words = len(re.findall(r'\b[A-Z][a-z]{2,}\b', transcript))
    entity_density = caps_words / max(duration_minutes, 1)
    entity_score = min(entity_density / 10.0, 1.0)

    # Combined transcript-based score
    score = (
        0.25 * vocab_score +      # Vocabulary richness
        0.20 * filler_score +      # Low filler content
        0.25 * factual_score +     # Factual claims density
        0.15 * pace_score +        # Speaking pace
        0.15 * entity_score        # Named entity density
    )

    return min(score, 1.0)


def compute_info_density_score(
    title: str,
    description: str,
    duration_seconds: int,
    transcript: Optional[str] = None,
) -> float:
    """
    Compute information density score ∈ [0, 1].
    Higher = more useful information per minute.

    When a transcript is available, it contributes 70% of the score
    (since it's the actual content). Description/title contribute 30%.

    Without a transcript, we rely entirely on title + description signals.
    """
    if duration_seconds <= 0:
        return 0.0

    duration_minutes = duration_seconds / 60

    # ── Description-based score (always computed) ──
    text = f"{title} {description}"
    factual_signals = _count_factual_signals(text)
    raw_density = factual_signals / max(duration_minutes, 1)
    structure = _has_structure(description)
    desc_words = len(description.split()) if description else 0
    desc_richness = min(desc_words / max(duration_minutes * 20, 1), 1.0)

    desc_score = (
        0.4 * min(raw_density / 5.0, 1.0) +
        0.3 * structure +
        0.3 * desc_richness
    )
    desc_score = min(desc_score, 1.0)

    # ── Transcript-based score (when available) ──
    if transcript and len(transcript.split()) > 50:
        transcript_score = _compute_transcript_density(transcript, duration_minutes)

        # Weighted blend: transcript is the primary signal
        final_score = 0.70 * transcript_score + 0.30 * desc_score
    else:
        # No transcript — rely on description only
        final_score = desc_score

    return round(min(final_score, 1.0), 4)
