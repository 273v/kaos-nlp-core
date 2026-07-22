# Readability scoring — design note and implementation plan

Status: **implemented** (all three phases). A `readability` module
providing Flesch Reading Ease, Flesch-Kincaid Grade, ARI, Coleman-Liau,
SMOG, Gunning Fog, Dale-Chall, plus LIX and RIX (nearly free given the
long-word count), following the layering of the `quality` module
(`rust/core/quality/` + `python/kaos_nlp_core/quality.py`).

## Why

Readability formulas need exactly the primitives this package already
owns — sentence segmentation (Punkt, Rust-backed), word tokenization,
character classification, and FST word lists — plus one new kernel: a
syllable estimator. Nothing in the ecosystem combines a fast Rust core
with honest formula provenance; the widely used Python option (textstat)
has drifted across versions as its syllable backend changed and silently
deviates from the literature on Gunning Fog and Dale-Chall.

## Design principles

Follow the `quality` split: **Rust computes text primitives** in one
GIL-released pass (tokens, letters, syllables, complex-word counts);
**Python owns the formula arithmetic and tunable constants** as
module-level tables, tunable without rebuilding the wheel. Sentence
counts reuse `PunktTokenizer.count_sentences`
(`rust/bindings/segmentation.rs`) — no new segmentation code.

Two research findings shape the data decisions:

1. **Syllables: heuristic first, CMUdict as the accuracy upgrade.**
   Hyphenation-pattern counting (Pyphen-style Knuth-Liang) undercounts
   by design: patterns favour precision over recall, and the
   `lefthyphenmin`/`righthyphenmin` convention means short words like
   "ago" or "oboe" get zero break points (~7% wrong counts in a
   published sample, arXiv 2102.08858). The accuracy ceiling among
   non-ML approaches is CMU Pronouncing Dictionary lookup with a
   heuristic fallback (textstat's current architecture). CMUdict is
   2-clause BSD (~134k entries) — clean to bundle with a NOTICE entry.
   A tuned vowel-group heuristic needs **zero new crates**: `regex`,
   `icu_properties`, and `rust/core/characters/` are already in-tree.
2. **Dale-Chall word list: do not bundle-and-relicense.** The 1995
   revised ~3,000-word list comes from a copyrighted book (Chall & Dale,
   *Readability Revisited*, 1995) with no licensed canonical source.
   textstat bundles it under a blanket MIT claim it arguably cannot
   make; koRpus refuses to ship it "for copyright reasons". Per the
   provenance rules in AGENTS.md, `dale_chall` computes only when the
   caller supplies a familiar-word `FstSet` (existing `matching.FstSet`
   infrastructure), with a build script converting any user-provided
   word list to FST.

## File footprint

Registration follows the fixed 4-point chain (`rust/core/mod.rs`,
`rust/bindings/mod.rs`, `rust/lib.rs`, plus the module files):

| File | Contents |
|---|---|
| `rust/core/readability/mod.rs` | `TextCounts` accumulator: words, letters, letters+digits, syllable total, polysyllable (≥3) count, Fog complex-word count, unfamiliar-word count (vs optional `FstSet`). Single pass over tokens; hard input-size limits. |
| `rust/core/readability/syllable.rs` | `estimate_syllables(word) -> u32`: lowercase/strip, vowel-group count, silent-e, `-le`/`-les` endings, non-syllabic `-es`/`-ed` handling, small embedded exception table. Deterministic; minimum 1. |
| `rust/core/mod.rs` | add `pub mod readability;` (alphabetical) |
| `rust/bindings/readability.rs` | `register_module` per the `quality.rs` pattern (including the `sys.modules` re-registration); `#[pyclass(frozen, get_all)] PyTextCounts`; `analyze(text, num_sentences, lexicon: Option<&PyFstSet>)` using `py.detach` and `PyFstSet::inner_ref()`; `syllable_count(word)` exposed for testing. |
| `rust/bindings/mod.rs`, `rust/lib.rs` | register the submodule |
| `python/kaos_nlp_core/readability.py` | Public wrapper: `compute_counts(text)`, `readability_report(text, *, familiar_words: FstSet \| None = None)`; frozen slotted dataclasses `TextCounts`, `ReadabilityScores` with `to_dict()` (None fields omitted); formula constants as module-level tables; sentence count via `get_default_punkt_tokenizer().count_sentences()`. |
| `python/kaos_nlp_core/_rust/readability.pyi` | stub mirroring the binding |
| `python/kaos_nlp_core/cli.py` | `readability` subcommand (`--json`, `--familiar-words PATH`), modeled on `_cmd_analyze` |
| `scripts/build_familiar_wordset.py` | text word list → FST, mirroring `scripts/build_default_wordset.py` |
| `tests/unit/test_readability.py`, `tests/unit/test_cli.py`, `tests/bench_readability.py` | see Testing |

## Formulas (verified constants and sources)

- **Flesch Reading Ease**: `206.835 − 1.015·(W/S) − 84.6·(syll/W)`
  (Flesch 1948, *A New Readability Yardstick*).
- **Flesch-Kincaid Grade**: `0.39·(W/S) + 11.8·(syll/W) − 15.59`
  (Kincaid et al. 1975, DTIC AD-A006655).
- **ARI**: `4.71·(chars/W) + 0.5·(W/S) − 21.43`; chars = **letters +
  digits only**; result rounded up to grade (Smith & Senter 1967,
  AMRL-TR-66-220).
- **Coleman-Liau**: `0.0588·L − 0.296·S − 15.8`, per-100-words published
  form, **letters only** (Coleman & Liau 1975).
- **SMOG**: `1.0430·√(polysyllables·30/S) + 3.1291` (McLaughlin 1969).
  Constants are calibrated to 30-sentence samples; return the score
  plus a `smog_valid` flag when `S < 30` rather than erroring
  (py-readability-metrics) or computing silently (textstat).
- **Gunning Fog**: `0.4·[(W/S) + 100·(complex/W)]` (Gunning 1952).
  Complex-word counting implements the mechanizable exclusions — do not
  count `-es`/`-ed`/`-ing` as the third syllable; skip
  mid-sentence-capitalized proper nouns; skip hyphenated compounds —
  each behind a config flag. textstat ignores all of these exclusions
  and will disagree; document that.
- **Dale-Chall**: `0.1579·PDW + 0.0496·ASL`, `+3.6365` when
  `PDW > 5%`; `None` when no lexicon supplied. This is the **1948
  regression** (what every major library actually implements); the 1995
  "New Dale-Chall" maps counts through cloze tables. Document the claim
  precisely: 1948 formula, usable with the 1995 word list the user
  supplies.

## Testing

- Hand-computed fixture paragraphs with exact expected counts and
  scores, tested through the public wrapper.
- Syllable-estimator accuracy test against a CMUdict-derived labeled
  sample committed as a small fixture (assert ≥ ~90% exact match, to
  pin regressions).
- Unicode boundary tests (CJK, emoji, mixed-script): formulas are
  English-calibrated; counting must be deterministic and panic-free on
  any input, and documented as such.
- proptest: `analyze` never panics on arbitrary UTF-8; syllables ≥ 1
  for any non-empty word.
- CLI `--json` golden test.
- Gates: `cargo fmt/clippy/test`, `ruff`, `ty`, `pytest`,
  `uv run maturin build --release`; `cargo audit` / `cargo deny check`
  when the Phase 2 data lands. No new crates in any phase.

## Phasing

1. **Core + formulas, heuristic syllables.** Everything above; all
   seven scores working (Dale-Chall lexicon-gated). No new deps, no
   new bundled data.
2. **CMUdict syllable map (accuracy upgrade).**
   `scripts/build_syllable_map.py` converting CMUdict to a
   word → count `fst::Map` in `python/kaos_nlp_core/data/` (~1 MB,
   alongside the existing 1.9 MB `english_wordset.fst`); BSD notice
   added to NOTICE; lookup-then-heuristic-fallback in `syllable.rs`.
3. **Benches + docs + changelog.** `tests/bench_readability.py`
   throughput bench (shakespeare/war_and_peace fixtures), optional
   criterion bench for the syllable kernel, README example, CHANGELOG
   entry.

## Resolved decisions (as built)

- **CMUdict map: bundled.** `python/kaos_nlp_core/data/cmudict_syllables.fst`
  came in at ~660 kB (below the 1 MB estimate); BSD notice added to
  NOTICE; built by `scripts/build_syllable_map.py` from first
  pronunciations (syllables = phonemes carrying stress digits —
  the deterministic multi-pronunciation rule).
- **Dale-Chall: user-supplied list** (the koRpus model), via
  `scripts/build_familiar_wordset.py` → `FstSet`. Proper nouns
  (title-case, not sentence-initial) count as familiar per the
  Dale-Chall procedure.
- **Gunning Fog: literature-faithful by default**, each exclusion
  individually flag-gated (`fog_exclude_suffixes`,
  `fog_exclude_proper_nouns`, `fog_exclude_compounds`); CLI
  `--naive-fog` disables all three for textstat-comparable numbers.
- **Word definition** (the biggest cross-library divergence source):
  a whitespace-delimited token, outer punctuation stripped by the
  shared tokenizer, containing ≥1 letter or digit. "don't" and
  "mother-in-law" are one word each; purely numeric tokens are words
  (1 syllable each; their digits count for ARI).
- **Sentence-start detection** (needed by the proper-noun exclusion
  without new segmentation): first word, previous raw chunk ending in
  `.`/`!`/`?`/`…` (ignoring trailing closing quotes/brackets), or a
  preceding blank line. Deterministic, documented, tested.
- **Zero-division policy:** no words or no sentences → every score is
  `None` (and `smog_valid=False`); `to_dict()` omits `None` fields.
- **ARI returns the raw float** (the dataclass stays uniformly
  `float | None`); the round-up-to-grade convention is documented.
- **Syllable heuristic accuracy:** 92.2% exact match on all ~125k
  CMUdict entries (roughly half of which are proper names); a
  committed 500-word deterministic sample
  (`tests/fixtures/syllables_cmudict_sample.tsv`) pins ≥90% in CI, and
  the bundled map scores 100% on the same sample.
- **Measured throughput** (1 MB Gutenberg text, single thread):
  Rust counting pass ~30 MB/s; full `readability_report` including
  Punkt sentence counting ~11.5 MB/s.
