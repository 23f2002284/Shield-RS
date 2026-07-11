"""
Module 1c: Social Proof Stripper

The simplest but most philosophically important module.
It strips all social proof signals from video metadata so the
downstream scorer never sees them.

This is the "algorithmic inversion" in action: YouTube BOOSTS
content based on view counts, like counts, and subscriber counts.
Our agent IGNORES them entirely.
"""

from typing import Dict, Any


def strip_social_proof(video: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of the video metadata with all social proof
    signals set to None. The downstream scorer never sees these.

    Stripped fields:
    - view_count: Popularity ≠ quality
    - like_count: Herd approval ≠ personal relevance
    - subscriber_count: Creator fame ≠ content accuracy
    """
    stripped = video.copy()
    stripped["view_count"] = None
    stripped["like_count"] = None
    stripped["subscriber_count"] = None
    return stripped


def strip_social_proof_batch(videos: list) -> list:
    """Strip social proof from a list of video metadata dicts."""
    return [strip_social_proof(v) for v in videos]
