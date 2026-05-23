from __future__ import annotations

from typing import Iterable, Optional

from .config import (
    DEFAULT_RATIO_SCORE_RULES,
    DEFAULT_SCORE_RULES,
    RatioRule,
    ScoreRule,
)


def pct_to_rule(
    pct: float,
    rules: Optional[Iterable[ScoreRule]] = None,
) -> ScoreRule:
    """Map a percentage to the highest matching score rule.

    Parameters
    ----------
    pct : float
        Percentage value to evaluate.
    rules : iterable of ScoreRule or None, optional
        Score rules to use. If ``None``, ``DEFAULT_SCORE_RULES`` are used.

    Returns
    -------
    ScoreRule
        Highest matching score rule. If no rules are provided, a neutral
        default rule is returned.
    """
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


def pct_to_score_verdict(
    pct: float,
    rules: Optional[Iterable[ScoreRule]] = None,
) -> tuple[int, str]:
    """Map a percentage to a numeric score and verdict.

    Parameters
    ----------
    pct : float
        Percentage value to evaluate.
    rules : iterable of ScoreRule or None, optional
        Score rules to use. If ``None``, ``DEFAULT_SCORE_RULES`` are used.

    Returns
    -------
    tuple of int and str
        Numeric score and verdict from the highest matching rule.
    """
    r = pct_to_rule(pct, rules=rules)
    return int(r.score), str(r.verdict)


def ratio_to_rule(
    ratio: float,
    rules: Optional[Iterable[RatioRule]] = None,
) -> RatioRule:
    """Map a ratio to the highest matching ratio rule.

    Parameters
    ----------
    ratio : float
        Ratio value to evaluate.
    rules : iterable of RatioRule or None, optional
        Ratio rules to use. If ``None``, ``DEFAULT_RATIO_SCORE_RULES`` are used.

    Returns
    -------
    RatioRule
        Highest matching ratio rule. If no rules are provided, a neutral
        default rule is returned.
    """
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


def ratio_to_score_verdict(
    ratio: float,
    rules: Optional[Iterable[RatioRule]] = None,
) -> tuple[int, str]:
    """Map a ratio to a numeric score and verdict.

    Parameters
    ----------
    ratio : float
        Ratio value to evaluate.
    rules : iterable of RatioRule or None, optional
        Ratio rules to use. If ``None``, ``DEFAULT_RATIO_SCORE_RULES`` are used.

    Returns
    -------
    tuple of int and str
        Numeric score and verdict from the highest matching rule.
    """
    r = ratio_to_rule(ratio, rules=rules)
    return int(r.score), str(r.verdict)
