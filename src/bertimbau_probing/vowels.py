"""Vowel-density utilities -- the probing target.

Pure Python (standard library only) so this is cheap to import and test.
"""
from __future__ import annotations

import re

# Portuguese letters, including accented forms.
_NON_LETTER = re.compile(r"[^A-Za-zÇçÃãÁáÉéÍíÓóÚúÀàÂâÊêÔôÕõÜü]")
_VOWEL = re.compile(r"[AEIOUaeiouÁÉÍÓÚáéíóúÀàÂâÃãÊêÍÔôÕõÜü]")

LOW, MID, HIGH = 0, 1, 2


def vowel_density(text: str) -> float:
    """Return the fraction of letters in ``text`` that are vowels.

    Non-letter characters are ignored. Returns ``0.0`` when there are no letters.
    """
    letters = _NON_LETTER.sub("", text or "")
    if not letters:
        return 0.0
    return len(_VOWEL.findall(letters)) / len(letters)


def density_band(density: float) -> int:
    """Bucket a density into three bands: low (<1/3), mid (1/3-2/3), high (>2/3)."""
    if density < 1 / 3:
        return LOW
    if density <= 2 / 3:
        return MID
    return HIGH


def first_word_density(text: str) -> float:
    """Vowel density of the first whitespace-delimited word."""
    words = (text or "").split()
    return vowel_density(words[0]) if words else 0.0


def last_word_density(text: str) -> float:
    """Vowel density of the last whitespace-delimited word."""
    words = (text or "").split()
    return vowel_density(words[-1]) if words else 0.0
