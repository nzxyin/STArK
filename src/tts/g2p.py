"""Text -> phoneme-id G2P for STArK inference.

Uses Phonemizer with the eSpeak NG backend (requires the `espeak-ng` system package — see
README's Installation section) to reproduce the phoneme convention `ipa.py`'s vocabulary and the
training data were built on:
  - words are whitespace-tokenized from espeak's IPA output (espeak itself sometimes glues
    adjacent function words with no space, e.g. "from the" -> "fɹʌmðə" — that's expected and
    matches the training data, which used the same tokenization)
  - both primary (ˈ) and secondary (ˌ) stress marks are rewritten as a standalone `*` token
    immediately before the stressed syllable's vowel
  - words are joined with `|`
  - trailing punctuation on a word is converted to a silence token and inserted between that
    word and the following `|`: `,;:—…` -> `<sp>`, `.` -> `<eos>`, `!` -> `<exclm>`,
    `?` -> `<quest>`; quotes/parens/other punctuation are dropped
  - if the text doesn't end in sentence-final punctuation, an `<eos>` is appended anyway (the
    model was always trained with a trailing silence token)

This was reverse-engineered from the training data (no G2P code shipped with the original
preprocessing pipeline) and is a best-effort reproduction, not guaranteed byte-identical to
whatever produced the original phn/ files for every possible input.
"""
import re

import numpy as np

from ipa import get_id, symbols as _ipa_symbols

_PUNCT_TO_TOKEN = {
    ",": "<sp>", ";": "<sp>", ":": "<sp>", "—": "<sp>", "…": "<sp>", "-": "<sp>",
    ".": "<eos>",
    "!": "<exclm>",
    "?": "<quest>",
}
_DROPPED_PUNCT = set('"\'()[]{}«»“”')

# Longest-symbol-first so multi-character phonemes (e.g. "eɪ", "tʃ", "n̩") match before their
# single-character prefixes do.
_PHONEME_SYMBOLS = sorted((s for s in _ipa_symbols if s not in ("_", "|", "*")
                            and not s.startswith("<")), key=len, reverse=True)

_backend = None


def _get_backend():
    global _backend
    if _backend is not None:
        return _backend
    from phonemizer.backend import EspeakBackend

    _backend = EspeakBackend(
        "en-us",
        preserve_punctuation=True,
        with_stress=True,
        punctuation_marks=';:,.!?¡¿—…"«»“”()-',
        language_switch="remove-flags",
    )
    return _backend


def _split_phonemes(word_ipa):
    """Split a raw espeak IPA word (with ˈ/ˌ stress marks still inline) into a list of
    ipa.py symbols, converting stress marks to standalone '*' tokens."""
    out = []
    i = 0
    n = len(word_ipa)
    while i < n:
        ch = word_ipa[i]
        if ch in ("ˈ", "ˌ"):  # primary / secondary stress
            out.append("*")
            i += 1
            continue
        matched = False
        for sym in _PHONEME_SYMBOLS:
            if not word_ipa.startswith(sym, i):
                continue
            # 'əl' (syllabic l, as in "little" -> l ɪ ɾ əl) only occurs word-finally in
            # practice; elsewhere "ə" + "l" are two independent phonemes that happen to be
            # adjacent (e.g. "hello" -> h ə l oʊ, not h [əl] oʊ).
            if sym == "əl" and i + len(sym) != n:
                continue
            out.append(sym)
            i += len(sym)
            matched = True
            break
        if not matched:
            # Unknown character (e.g. an allophone not in ipa.py's vocab) — skip it rather
            # than crash; this can degrade quality for unusual input but keeps inference usable.
            i += 1
    return out


def text_to_phonemes(text):
    """Returns a list of ipa.py phoneme-vocabulary symbols, e.g.
    ['ʃ', 'iː', '|', 'k', '*', 'ɛ', 'p', 't', ..., '<eos>']."""
    backend = _get_backend()
    raw = backend.phonemize([text], strip=False)[0]

    symbols = []
    for raw_word in raw.split():
        # Pull off trailing punctuation (espeak attaches it directly to the preceding word).
        m = re.match(r"^(.*?)([,;:.!?—…]*)$", raw_word)
        core, trailing_punct = m.group(1), m.group(2)
        core = "".join(ch for ch in core if ch not in _DROPPED_PUNCT)
        if not core:
            continue
        symbols.extend(_split_phonemes(core))
        symbols.append("|")
        for ch in trailing_punct:
            token = _PUNCT_TO_TOKEN.get(ch)
            if token:
                # Replace the '|' just added with <token> |, matching the training convention
                # of "<sp> |" / "<eos> |" rather than "| <sp>".
                symbols[-1] = token
                symbols.append("|")

    if symbols and symbols[-1] == "|":
        symbols.pop()
    if not symbols or symbols[-1] not in ("<eos>", "<exclm>", "<quest>"):
        symbols.append("<eos>")

    return symbols


def text_to_phoneme_ids(text):
    """Returns (phoneme_symbols, np.ndarray of ids) ready for TTS.forward's `phones` input."""
    symbols = text_to_phonemes(text)
    ids = np.array([get_id(s) for s in symbols], dtype=np.int64)
    return symbols, ids
