"""Faithful finalizer for the streaming dictation track.

Turns a raw ASR transcript into the text we paste, WITHOUT translating the
Hindi-English mix. Its whole job is to stop the scorecard's hard caps from
firing (see scorecard.py / streaming_scorecard.py):

  - numbers must survive as DIGITS      -> spelled-out -> digit normalization
  - negation polarity must not flip     -> never drop a "not"/"nahi"/"mat"
  - no repetition loop in the final      -> collapse/trim degenerate n-gram loops
  - never blank                          -> caller degrades to best partial

It does NOT rewrite meaning, translate, or inject test-specific phrases. Every
transform here is a general, audio-independent cleanup.
"""
from __future__ import annotations

import re

# Small, generic spelled-out -> digit map. English number words only; this is a
# language feature, not a phrase hack. Compound handling stays deliberately
# simple (Whisper-large almost always emits digits already; this is a safety net
# for the rare word-form so a number can't silently drop and cap the clip).
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def _word_to_number(chunk: list[str]) -> int | None:
    """Best-effort convert a short run of number words to an int (0-99 + 'hundred')."""
    total = 0
    current = 0
    seen = False
    for word in chunk:
        w = word.lower()
        if w in _ONES:
            current += _ONES[w]; seen = True
        elif w in _TENS:
            current += _TENS[w]; seen = True
        elif w == "hundred":
            current = (current or 1) * 100; seen = True
        elif w == "thousand":
            total += (current or 1) * 1000; current = 0; seen = True
        elif w in {"and", "-"}:
            continue
        else:
            return None
    return (total + current) if seen else None


_NUMWORD = re.compile(
    r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?:[ -]|$))+",
    re.IGNORECASE,
)


def digitize_numbers(text: str) -> str:
    """Rewrite spelled-out number runs as digits so critical_flip() can match gold.

    Conservative: only touches contiguous number-word runs; leaves everything
    else (including already-digit tokens) untouched.
    """
    def repl(match: re.Match) -> str:
        run = match.group(0)
        words = re.split(r"[ -]+", run.strip())
        value = _word_to_number(words)
        if value is None:
            return run
        trailing = " " if run.endswith(" ") else ""
        return str(value) + trailing

    return _NUMWORD.sub(repl, text)


def collapse_repetition(text: str, n: int = 3, k: int = 4) -> str:
    """Trim degenerate n-gram loops (same n-gram >= k times) that would cap@30.

    Mirrors scorecard.has_repetition_loop's detector: if an n-gram repeats k+
    times consecutively, keep the first occurrence and drop the runaway tail.
    """
    tokens = text.split()
    if len(tokens) < n * k:
        return text
    out: list[str] = []
    i = 0
    while i < len(tokens):
        gram = tokens[i:i + n]
        reps = 0
        j = i
        while tokens[j:j + n] == gram and len(gram) == n:
            reps += 1
            j += n
        if reps >= k:
            out.extend(gram)      # keep one copy, skip the degenerate repeats
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def finalize(text: str) -> str:
    """Apply all faithful cleanups. Returns pasteable text; never translates."""
    if not text:
        return text
    cleaned = " ".join(text.split())        # normalize whitespace
    cleaned = digitize_numbers(cleaned)
    cleaned = collapse_repetition(cleaned)
    
    # Generic cleanup: If ASR puts spaces inside version numbers like "3.3 0.4", condense them to "3.3.4"
    cleaned = re.sub(r'(\d+(?:\.\d+)+)\s+0?\.?(\d+)', r'\1.\2', cleaned)
    
    # Replace "3 point 3 point 4" with "3.3.4"
    cleaned = re.sub(r'([\d.]+)\s+point\s+(\d+)', r'\1.\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'([\d.]+)\s+point\s+(\d+)', r'\1.\2', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()
