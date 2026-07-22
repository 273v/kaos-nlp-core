"""Type stubs for kaos_nlp_core._rust.readability."""

from typing import Any

class SyllableMap:
    def __init__(self, entries: list[tuple[str, int]]) -> None: ...
    @staticmethod
    def load(path: str) -> SyllableMap: ...
    def save(self, path: str) -> None: ...
    def get(self, word: str) -> int | None: ...
    def __len__(self) -> int: ...
    def __contains__(self, word: str) -> bool: ...

class TextCounts:
    words: int
    letters: int
    letters_and_digits: int
    syllables: int
    polysyllable_words: int
    fog_complex_words: int
    long_words: int
    unfamiliar_words: int | None

def analyze(
    text: str,
    lexicon: Any | None = None,
    syllable_map: SyllableMap | None = None,
    fog_exclude_suffixes: bool = True,
    fog_exclude_proper_nouns: bool = True,
    fog_exclude_compounds: bool = True,
) -> TextCounts: ...
def syllable_count(word: str, syllable_map: SyllableMap | None = None) -> int: ...
