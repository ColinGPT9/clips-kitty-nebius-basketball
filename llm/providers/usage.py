"""The tokens each cloud model was billed for, as its provider reported them.

Every chat completion names its usage (prompt, completion, and where the
provider says so, cached and reasoning tokens). They are added up here per
"<provider>/<model>", for the job log and for docs/hackathon/BENCHMARKS.md.
Only counts are kept: never a prompt, an answer or a key. The provider bills
the user's own account; this is what it said, not a bill.
"""

import threading

_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")
_lock = threading.Lock()
_totals: dict[str, dict[str, int]] = {}


def record(provider: str, model: str, data: dict) -> None:
    """Add one response's `usage` ({} or absent: counted as a call only)."""
    usage = data.get("usage") if isinstance(data, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    row = {
        "calls": 1,
        "prompt_tokens": _int(usage.get("prompt_tokens")),
        "completion_tokens": _int(usage.get("completion_tokens")),
        "cached_tokens": _int(prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else 0),
        "reasoning_tokens": _int(completion_details.get("reasoning_tokens")
                                 if isinstance(completion_details, dict) else 0),
    }
    with _lock:
        total = _totals.setdefault(f"{provider}/{model}", dict.fromkeys(_FIELDS, 0))
        for field in _FIELDS:
            total[field] += row[field]


def totals() -> dict[str, dict[str, int]]:
    with _lock:
        return {name: dict(row) for name, row in _totals.items()}


def diff(before: dict, after: dict) -> dict[str, dict[str, int]]:
    """What was added between two totals(), leaving out models with nothing new."""
    out = {}
    for name, row in after.items():
        old = before.get(name) or {}
        delta = {field: row.get(field, 0) - old.get(field, 0) for field in _FIELDS}
        if any(delta.values()):
            out[name] = delta
    return out


def summary(rows: dict[str, dict[str, int]]) -> str:
    """One line per model, for the job log."""
    parts = []
    for name, row in sorted(rows.items()):
        line = (f"{name}: {row['calls']} call(s), {row['prompt_tokens']:,} tokens in, "
                f"{row['completion_tokens']:,} out")
        if row.get("reasoning_tokens"):
            line += f" ({row['reasoning_tokens']:,} reasoning)"
        parts.append(line)
    return "; ".join(parts)


def reset() -> None:
    with _lock:
        _totals.clear()


def _int(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):  # Infinity, NaN
        return 0
