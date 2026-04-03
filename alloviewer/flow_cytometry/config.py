from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ScoreRule:
    """Rule: if pct >= min_pct, apply this rule (highest matching min_pct wins)."""
    min_pct: float
    score: int
    verdict: str

@dataclass(frozen=True)
class RatioRule:
    """Rule: if ratio >= min_ratio, apply this rule (highest matching min_ratio wins)."""
    min_ratio: float
    score: int
    verdict: str

DEFAULT_SCORE_RULES: List[ScoreRule] = [
    ScoreRule(min_pct=0.0,  score=0, verdict="Negative"),
    ScoreRule(min_pct=5.0,  score=1, verdict="Low"),
    ScoreRule(min_pct=20.0,  score=2, verdict="Moderate"),
    ScoreRule(min_pct=60.0, score=3, verdict="Strong"),
]

DEFAULT_RATIO_SCORE_RULES: List[RatioRule] = [
    RatioRule(min_ratio=1.0, score=0, verdict="Negative"),
    RatioRule(min_ratio=2.0, score=1, verdict="Low"),
    RatioRule(min_ratio=4.0, score=2, verdict="Moderate"),
    RatioRule(min_ratio=6.0, score=3, verdict="Strong"),
]
