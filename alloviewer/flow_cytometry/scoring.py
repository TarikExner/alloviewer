from __future__ import annotations
from typing import Iterable, Optional
from .config import ScoreRule, RatioRule, DEFAULT_RATIO_SCORE_RULES, DEFAULT_SCORE_RULES


def pct_to_rule(pct: float, rules: Optional[Iterable[ScoreRule]] = None) -> ScoreRule:
    rlist = list(rules) if rules is not None else list(DEFAULT_SCORE_RULES)
    if not rlist:
        return ScoreRule(min_pct=0.0, score=0, verdict="")
    rlist = sorted(rlist, key=lambda r: float(r.min_pct))

    p = float(pct)
    chosen = rlist[0]
    for r in rlist:
        if p >= float(r.min_pct):
            chosen = r
    return chosen


def pct_to_score_verdict(pct: float, rules: Optional[Iterable[ScoreRule]] = None) -> tuple[int, str]:
    r = pct_to_rule(pct, rules=rules)
    return int(r.score), str(r.verdict)


def ratio_to_rule(ratio: float, rules: Optional[Iterable[RatioRule]] = None) -> RatioRule:
    rlist = list(rules) if rules is not None else list(DEFAULT_RATIO_SCORE_RULES)
    if not rlist:
        return RatioRule(min_ratio=0.0, score=0, verdict="")
    rlist = sorted(rlist, key=lambda r: float(r.min_ratio))

    x = float(ratio)
    chosen = rlist[0]
    for r in rlist:
        if x >= float(r.min_ratio):
            chosen = r
    return chosen


def ratio_to_score_verdict(ratio: float, rules: Optional[Iterable[RatioRule]] = None) -> tuple[int, str]:
    r = ratio_to_rule(ratio, rules=rules)
    return int(r.score), str(r.verdict)

