import math

from bertimbau_probing.vowels import (
    HIGH,
    LOW,
    MID,
    density_band,
    first_word_density,
    last_word_density,
    vowel_density,
)


def test_vowel_density_basic():
    # "casa" -> a, a vowels of 4 letters
    assert vowel_density("casa") == 0.5
    # all vowels
    assert vowel_density("aeiou") == 1.0
    # no vowels
    assert vowel_density("xyz") == 0.0


def test_vowel_density_ignores_non_letters_and_handles_empty():
    assert vowel_density("a1!a") == 1.0  # digits/punctuation ignored, both letters vowels
    assert vowel_density("") == 0.0
    assert vowel_density("12345") == 0.0


def test_vowel_density_counts_accented_vowels():
    # "ótima": ó,i,a vowels (3) of 5 letters
    assert math.isclose(vowel_density("ótima"), 3 / 5)


def test_density_band_thresholds():
    assert density_band(0.2) == LOW
    assert density_band(0.5) == MID
    assert density_band(1 / 3) == MID      # boundary is inclusive of mid
    assert density_band(2 / 3) == MID
    assert density_band(0.9) == HIGH


def test_word_baselines():
    assert first_word_density("casa xyz") == 0.5
    assert last_word_density("xyz casa") == 0.5
    assert first_word_density("") == 0.0
    assert last_word_density("   ") == 0.0
