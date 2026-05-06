# cuad-sample — provenance

5 contracts + 1 golden JSONL extracted from the **Contract Understanding
Atticus Dataset (CUAD) v1**. Used by the alpha-extractor regression
suite (`tests/extract/alpha/test_alpha_*.py`) as hypothesis falsifiers.

For full attribution and citation see [DATA_LICENSES.md](DATA_LICENSES.md);
the per-file mapping back to CUAD indices and SHA-256s lives in
[MANIFEST.json](MANIFEST.json). This README is the policy-required
columnar provenance table per
`docs/oss/50-data-and-fixtures/provenance-policy.md`.

## Per-file provenance

| File | Source URL | License | Retrieved | SHA-256 |
|---|---|---|---|---|
| `centrackinternationalinc-10-29-1999-ex-10-3-web-site-hosting.txt` | CUAD v1, contract index 3 (`CENTRACKINTERNATIONALINC_10_29_1999-EX-10.3-WEB SITE HOSTING AGREEMENT`) — <https://github.com/TheAtticusProject/cuad> | CC-BY-4.0 | 2026-04-08 | `8532356d811b76bd7ae536cf22910742ba61815dc39741fd9b295d7bc80c99f1` |
| `dragonsystemsinc-01-08-1999-ex-10-17-outsourcing-agreement.txt` | CUAD v1, contract index 62 — <https://github.com/TheAtticusProject/cuad> | CC-BY-4.0 | 2026-04-08 | `8d8020e5c0bcaafac609fe53a2e7edbe2ecbf15541dc846ed25ac63ec822b37e` |
| `lucidinc-04-15-2011-ex-10-9-distributor-agreement.txt` | CUAD v1, contract index 164 — <https://github.com/TheAtticusProject/cuad> | CC-BY-4.0 | 2026-04-08 | `85676ae23975c26b41b9bc868f278c17addb353cf956628ad3c6c6d6108ce716` |
| `mphasetechnologiesinc-20030911-10-k-ex-10-15-1560667-ex-10-1.txt` | CUAD v1, contract index 186 — <https://github.com/TheAtticusProject/cuad> | CC-BY-4.0 | 2026-04-08 | `c00e73fa23804a8ea70f011dfaca0debf75bccdeb375ca864cbf2a36a0ff4ce8` |
| `ticketscominc-06-22-1999-ex-10-22-sponsorship-agreement.txt` | CUAD v1, contract index 346 — <https://github.com/TheAtticusProject/cuad> | CC-BY-4.0 | 2026-04-08 | `4928e56782c0c6c8d589f9ceb019dc418e38b7b324be28f49a20a81b03d1eaa7` |
| `cuad-extraction-golden.jsonl` | Extracted from CUAD v1 SQuAD-format annotations (5 contracts × 5 target clauses) | CC-BY-4.0 (derived) | 2026-04-08 | `af1bed6bbfdbc1f8aad2a30daf7cea576605bf556fbc17df6245a020949b8342` |
| `DATA_LICENSES.md` | Hand-written attribution + license summary for this directory | Apache-2.0 (273V) | n/a (in-tree doc) | n/a |
| `MANIFEST.json` | Hand-written index linking each file to its CUAD source row | Apache-2.0 (273V) | n/a (in-tree doc) | n/a |

## CUAD upstream pin

| Field | Value |
|---|---|
| Repository | <https://github.com/TheAtticusProject/cuad> |
| Source artifact | `CUADv1.json` inside `data.zip` |
| Source archive SHA-256 | `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae` |
| DOI | <https://doi.org/10.5281/zenodo.4595826> |
| Citation | Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*. NeurIPS 2021 Datasets & Benchmarks Track. arXiv:2103.06268. |
| License | Creative Commons Attribution 4.0 International (CC-BY-4.0) |

## Modifications

Per DATA_LICENSES.md: contract `*.txt` files contain the **verbatim**
`context` field from CUAD with no text changes. Modifications applied:

1. Renamed files to lowercase-hyphenated slugs derived from CUAD titles.
2. Grouped CUAD's SQuAD-format spans by clause type into one JSON object
   per contract in JSONL.
3. Excluded the 36 clause types not targeted by this sample.

The redistributed subset is small (five 8–16 KB contracts, ~50 KB
total). Full evaluation runs require the upstream dataset; nothing in
this sample is a customer document or contains real PII beyond what is
already in the SEC EDGAR public record from which CUAD was sourced.
