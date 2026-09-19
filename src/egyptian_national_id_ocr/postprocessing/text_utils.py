"""Shared string-matching utilities. levenshtein() was previously duplicated
inline in tests/benchmark_accuracy.py; moved here so enum_matcher.py can use
the same implementation instead of a second copy drifting out of sync."""


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def char_accuracy(expected: str, actual: str) -> float:
    if not expected:
        return 1.0 if not actual else 0.0
    dist = levenshtein(expected, actual or "")
    return max(0.0, 1.0 - dist / len(expected))
