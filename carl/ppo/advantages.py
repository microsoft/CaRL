"""SMDP segment advantages (paper Eq. 1 + Appendix A).

Given a trajectory of N segments, with pre-segment boundary states s_0..s_{N-1}
having critic values v[0..N-1] and a binary terminal reward R:

    A_k = v[k+1] - v[k]    for k = 0..N-2     (intermediate)
    A_{N-1} = R - v[N-1]                       (terminal commits to R)

Sum telescopes to R - v[0] (Appendix A, Eq. 6); used as a sanity check.
"""
from __future__ import annotations
import math


def segment_advantages(values: list[float], reward: float) -> list[float]:
    """values = [V(s_0), ..., V(s_{N-1})] for an N-segment trajectory.
    Returns N advantages [A_0, ..., A_{N-1}]. Last advantage uses R, not V(s_N).
    """
    if not values:
        raise ValueError("Segment advantages require at least one boundary value")
    if not math.isfinite(reward) or any(not math.isfinite(value) for value in values):
        raise ValueError("Boundary values and terminal reward must be finite")
    A = [values[k + 1] - values[k] for k in range(len(values) - 1)]
    A.append(reward - values[-1])
    return A
