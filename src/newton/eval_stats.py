"""Oracle-evaluation counters. Reset before a run and read afterward."""

from collections import defaultdict

_STATS = defaultdict(int)
_ENABLED = True


def reset_eval_stats():
    global _STATS
    _STATS = defaultdict(int)


def get_eval_stats():
    return dict(_STATS)


def bump(key, n=1):
    if _ENABLED:
        _STATS[key] += n
