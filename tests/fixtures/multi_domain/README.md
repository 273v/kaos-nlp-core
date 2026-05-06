# Multi-domain validation corpus (P7.0c)

Hand-labeled fixtures for the structure-layer (P7) generality contract
(G1-G8) in `docs/SECTION_HEADING_PRIMITIVES_RESEARCH.md`.

## Layout

Each `*.txt` fixture has a sibling `*.gold.jsonl`. The gold file has one
JSON record per physical line of the fixture, in source order, with this
shape:

```json
{"line_index": 0, "label": "heading", "hierarchy_level": 1, "enumerator_kind": null}
```

Valid `label` values: `blank`, `heading`, `body`, `list_item`,
`table_row`, `metadata`, `boilerplate`. `hierarchy_level` is an integer
≥ 1 or `null`. `enumerator_kind` is one of the kind names from P3
(`decimal`, `roman_upper`, `alpha_lower`, `paren_alpha`, `section_word`,
`chapter_word`, `subpart_word`, etc.) or `null`.

## Domains covered

| Fixture | Domain | Lexicon (enum/hier/heading) |
|---|---|---|
| `academic_imrad.txt` | Academic paper (IMRAD) | english_legal_us / english_legal_us / english_academic |
| `rfc_dotted.txt` | RFC / W3C spec | english_legal_us / english_legal_us / none |
| `software_readme.txt` | Markdown software README | markdown_atx / markdown_atx / english_software |
| `wikipedia_short.txt` | News / Wikipedia | english_legal_us / english_legal_us / none |
| `de_bgb_section.txt` | German legal | german_legal / german_legal / german_legal |
| `fr_civil_section.txt` | French legal | french_legal / french_legal / french_legal |
| `es_civil_section.txt` | Spanish legal | spanish_legal / spanish_legal / spanish_legal |
| `financial_10k_excerpt.txt` | Financial report (10-K shape) | english_legal_us / english_legal_us / english_legal_us |
| `form_irs_excerpt.txt` | Form (label/value) | english_legal_us / none / none |
| `gutenberg_chapter.txt` | Literature | english_legal_us / english_legal_us / none |

## Coverage caveat

The design reference calls for ≥3 samples per domain (~30 fixtures
total) for G7 calibration; this seed has 1 per domain. The validation
script reports per-domain numbers and any sub-bound results triggers a
follow-up to enlarge the corpus.

## Per-file provenance

Per `docs/oss/50-data-and-fixtures/provenance-policy.md`. Every `*.txt`
slice was extracted via `scripts/build_multi_domain_corpus.py` from the
upstream sources documented below and **not modified afterwards**;
re-running the build script reproduces the exact byte content. Every
`*.gold.jsonl` was hand-labeled by 273V to ground-truth the structure
classifier (P7) — these are intellectual work product, not derived from
the upstream texts beyond the line indexing.

`Retrieved` = date the source was sampled into this corpus
(`build_multi_domain_corpus.py` first commit). All sampled rows are
pinned by upstream JSONL/PDF/DOCX index so re-running reproduces them
deterministically.

| File | Source URL | License | Retrieved | SHA-256 |
|---|---|---|---|---|
| `academic_imrad.txt` | Hand-crafted by 273V (synthetic IMRAD academic-paper structure) | Apache-2.0 (273V) | 2026-04-12 | `b4cc6559b6f1303d33a416df0a7f6e2d818c2b1b09e18d369280e7d19cd4f954` |
| `de_bgb_section.txt` | Bürgerliches Gesetzbuch (German Civil Code), official BMJV portal — <https://www.gesetze-im-internet.de/bgb/> | Public domain (German federal statute, §5 UrhG) | 2026-04-12 | `2bdd49078aa6b5d17c67be1f7425707607dd219dc10a5514b59c3642ba026e46` |
| `docx_cheese_curriculum.txt` | Sliced from `kaos-office/tests/fixtures/docx/CheeseSample.docx` via kaos-office | See sibling `kaos-office/tests/fixtures/docx/` provenance (public-domain USDA curriculum sample) | 2026-04-12 | `0adf4d7e45cd2c51e67dbb6e78b262c63fb72756046286d9064803181033cc82` |
| `docx_consumer_rights.txt` | Sliced from `kaos-office/tests/fixtures/docx/bcfp_consumer-rights-summary_2018-09.docx` via kaos-office | Public domain (CFPB government work) | 2026-04-12 | `e0cb6dbe34ddfcab09c53c376ceb2e3894c355fb88650c8182ea15a2cc3178b8` |
| `docx_form_intervention_planning.txt` | Sliced from `kaos-office/tests/fixtures/docx/Burnout_Intervention_Planning_Guide_Fillable_Form_1.docx` via kaos-office | Public domain (US federal-agency work) | 2026-04-12 | `c00e74eeff5e046c5618047ca622256b0f889cf8d852db1a3cbdd2b7f8313fb8` |
| `docx_multiparagraph_sample.txt` | Sliced from `kaos-office/tests/fixtures/docx/MultiParagraphSample.docx` via kaos-office | See sibling `kaos-office/tests/fixtures/docx/` provenance | 2026-04-12 | `2e25a66483424eeafef652cd4f40620b1cfec6bc892ad11bcb4450b54062770a` |
| `docx_policy_template.txt` | Sliced from `kaos-office/tests/fixtures/docx/PolicyProcedureTemplate_PhysicalFacility_Final.docx` via kaos-office | Public domain (US federal-agency template) | 2026-04-12 | `f2398644975a552c26ef25b2de4cbca2bc9f82e728c1ec77d7a2b49475d339c3` |
| `edgar_agreement_002.txt` | SEC EDGAR public filing, sampled from `edgar_agreements.jsonl` index 1 (HF dataset `alea-institute/kl3m-data-edgar-agreements`) | Public record (SEC filing, 17 USC §105 / SEC public availability) | 2026-04-12 | `bedd6779b23fef0e3d9bfd573e76e4b641496cbdbddbeafd1ad02cc6eb39aab3` |
| `edgar_agreement_003.txt` | SEC EDGAR public filing, sampled from `edgar_agreements.jsonl` index 5 | Public record (SEC filing) | 2026-04-12 | `d8bc4fcb2ed5ecfc190e2b707e4cbb98dc3e81b94b515bf0c46c10960cd55974` |
| `edgar_real_estate_purchase.txt` | SEC EDGAR public filing, sampled from `edgar_agreements.jsonl` index 0 | Public record (SEC filing) | 2026-04-12 | `a18f86d5acbb8c576fda7003174986f223e30aaff6fef31377b22819162f7b3e` |
| `fr_civil_section.txt` | Code civil français, official Légifrance — <https://www.legifrance.gouv.fr/codes/texte_lc/LEGITEXT000006070721/> | Open licence (Etalab v2) — French public-sector open data | 2026-04-12 | `1171c4317cd79d274b8c36f455fad88c712e82bab7baa2b9450f4ee31294bf55` |
| `gutenberg_shakespeare_play_list.txt` | Project Gutenberg #100 (Shakespeare complete works, table-of-contents slice) — <https://www.gutenberg.org/cache/epub/100/pg100.txt> | Public domain (US, ≥70 yrs after author's death) | 2026-04-12 | `45b596a8df35e367ac9d64afdf658191e9e8221f9de476494d3f8ed5ca7fcc73` |
| `gutenberg_shakespeare_sonnets.txt` | Project Gutenberg #100 (Shakespeare sonnets slice) — same upstream text as `gutenberg_shakespeare_play_list.txt` (deduplicated by SHA) | Public domain (US) | 2026-04-12 | `45b596a8df35e367ac9d64afdf658191e9e8221f9de476494d3f8ed5ca7fcc73` |
| `gutenberg_war_peace_book1ch1.txt` | Project Gutenberg #2600 (Tolstoy, *War and Peace*, Book 1 Ch 1 slice) — <https://www.gutenberg.org/cache/epub/2600/pg2600.txt> | Public domain (US) | 2026-04-12 | `bd9c60028a8d620d3d7c0741f1d3791d71912fe7335a6b8034e3ec26c622b200` |
| `gutenberg_war_peace_prose.txt` | Project Gutenberg #2600 (W&P deeper prose slice) | Public domain (US) | 2026-04-12 | `90784626d4a4b99196e5ac2828870570ec4461589e4233c03e6f637787bfa52d` |
| `gutenberg_war_peace_toc.txt` | Project Gutenberg #2600 (W&P chapter-listing TOC slice — same SHA as Book 1 Ch 1 because the TOC re-uses chapter-heading lines) | Public domain (US) | 2026-04-12 | `bd9c60028a8d620d3d7c0741f1d3791d71912fe7335a6b8034e3ec26c622b200` |
| `patent_001.txt` | USPTO patent, sampled from `patents.jsonl` index 0 (HF dataset `alea-institute/kl3m-data-patents`) | Public domain (US Government work, 17 USC §105) | 2026-04-12 | `75809eb1b0fd7fe53d5ef5e7e64c9c3453d4503c13cbf363082dd5f9d328065f` |
| `patent_002.txt` | USPTO patent, sampled from `patents.jsonl` index 5 | Public domain (US) | 2026-04-12 | `89a78b516b3de457428aa024dece5f675dc7bc27a080e5746e760e2d63cae34a` |
| `pdf_casd_court_order.txt` | Sliced from `kaos-pdf/tests/fixtures/casd_court_order.pdf` via kaos-pdf (US District Court, Southern District of California — public court order) | Public domain (US Government work, 17 USC §105) | 2026-04-12 | `01c8f89eef9397601459d957f12473a0ef34e433e9d6d2ef4996f828343e8b4d` |
| `pdf_fda_guidance_federal_register.txt` | Sliced from `kaos-pdf/tests/fixtures/kl3m_fda_guidance.pdf` via kaos-pdf (FDA guidance document, Federal Register publication) | Public domain (US) | 2026-04-12 | `d48a9e7ed9050337a6a83ffe41528c21bc1671366313a792b31dbb7be63507f0` |
| `pdf_staten_v_us_court_order.txt` | Sliced from `kaos-pdf/tests/fixtures/staten_v_united_states.pdf` via kaos-pdf (US federal court order) | Public domain (US) | 2026-04-12 | `2412d0b27ed530b6dbddf447e0f1c465cd4eb4b0201f2d68cd4e1444acff30f7` |
| `software_readme.txt` | Hand-crafted by 273V (synthetic Markdown software README structure) | Apache-2.0 (273V) | 2026-04-12 | `74201af7c17e150752a26351a0cd2fd31fac8f882cec42d8c18ecce6f7db9a2a` |
| `usc_ch15_military_support.txt` | US Code, sampled from `usc.jsonl` index 0 (HF dataset `alea-institute/kl3m-data-usc`) | Public domain (US federal statute) | 2026-04-12 | `9c439f9c5e5b23933ba2b505546f962da02f7edb1838d246cbc43f6a1705e9d5` |
| `usc_ch20_humanitarian.txt` | US Code, sampled from `usc.jsonl` index 20 | Public domain (US) | 2026-04-12 | `0136a06f075e46635f3d37315f13bc98abdcf1ac6abb46014ae5eff832995c05` |
| `usc_ch23_misc.txt` | US Code, sampled from `usc.jsonl` index 28 | Public domain (US) | 2026-04-12 | `88d1bfa23719484ec3c92dd7cfeb15330465bd795849cbc5c33a84c85892d738` |
| `wikipedia_short.txt` | Hand-crafted by 273V (Wikipedia-shape news/encyclopedia synthetic structure) | Apache-2.0 (273V) | 2026-04-12 | `50413fa057f5b7b9de010a1f238c5187cd6d61e72478cbe430fe47634b976810` |

### Hand-labeled gold annotations

Each `<slug>.gold.jsonl` is the per-line ground truth for the
corresponding `<slug>.txt`. Hand-labeled by 273V (no upstream source);
licensed Apache-2.0. Schema: `{"line_index": int, "label":
"blank|heading|body|list_item|table_row|metadata|boilerplate",
"hierarchy_level": int|null, "enumerator_kind": string|null}`.

| File | License | Retrieved | SHA-256 |
|---|---|---|---|
| `academic_imrad.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `606c8f9daf211b37684e3c708509a1115f09cd41f0d4a5f113af6865d89ed672` |
| `de_bgb_section.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `8ffa58cd1be324b21f3ded6d6030718ed7927d00623647d7b0630a375e979dd7` |
| `docx_cheese_curriculum.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `44e750178b3e02a78054c16c9ae0ee67f0d91333aae86067a792f9d9268e9370` |
| `docx_consumer_rights.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `a22f5d9c54dc7c3098deca576e2e25266db8eb0fdbbe58987fe5c42ae7103857` |
| `docx_form_intervention_planning.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `5e9815a24311aeaab72eff281fcc8de8129a2d66f3712f76dd1d9e01bc881a28` |
| `docx_multiparagraph_sample.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `3923b4de4eee44cb7247d1f80e34fd19898e238f1e7a45ee862243cd1f8feb58` |
| `docx_policy_template.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `4d5a911c3d37c132a1bee700fa3d2fb390aa9769cc829edfc1e98b1c1b803e25` |
| `edgar_agreement_002.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `2d5ff012b5256d6a2c3a3d0e0183232dabc3075cf7870a69ab519de8e70f9ef6` |
| `edgar_agreement_003.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `25e71a70c345ba6255d567f1b2252cb06ce8545c365dbda58652e56e161049af` |
| `edgar_real_estate_purchase.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `97c18cea099accd6b04173ff1c934b9a5abe6eff61b966c9af48eeb564c537c6` |
| `fr_civil_section.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `345d3340c8fa374f1dbfb02db3ecf1e378ea89d5e3e8cde54f9289924d0b0d69` |
| `gutenberg_shakespeare_sonnets.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `62a9bde51d7544cfc2f30c2beaa49a91c896bb0442ff1ca6a61702162e9b67c6` |
| `gutenberg_war_peace_book1ch1.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `501804f34924be1a5c98b3b8bfb38ba7cdd3cee601d8fdf1fd3304be5d0cbe91` |
| `gutenberg_war_peace_prose.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `773919db04f84231085d9dee59bdf57a1cc9d8c98f3f6cb53cbe9d75fac14198` |
| `patent_001.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `a69f3e08a5600611cbae810eda9a70acc2dbffc34204786542d5ff47c117a340` |
| `patent_002.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `c032651d6cb975f2128980d019e210410157b80eef68c28bb8c1ccc41b3b5dca` |
| `pdf_casd_court_order.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `3b4d68c9e17283aa45739144c07b430113eaf7f38e10f78783733d5eff1aacd5` |
| `pdf_fda_guidance_federal_register.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `467af521fdf149fa8c6ab330bd7070b9843e764ca6234c3aec7d7604cb88d515` |
| `pdf_staten_v_us_court_order.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `9f7123fe953f97a214c009ea9a65d5d01efd7483517dab5dde7612a464c6a474` |
| `software_readme.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `5be98513cc05b47977b7173a2ddc9c358d39083e0087d00367675fa8c81719ef` |
| `usc_ch15_military_support.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `54d591366838e3c85f760f9c908b1ba05becf926cf6d212743d69fdfdbfd23ac` |
| `usc_ch20_humanitarian.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `28b85292fc6cf0978fe90684396c78d913fdcae41528572853a90fce167870cb` |
| `usc_ch23_misc.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `d028846f01ef8e96480db175493675ee2fe2f4c915911a4aa7afbe9b58f1dc46` |
| `wikipedia_short.gold.jsonl` | Apache-2.0 (273V) | 2026-04-12 | `f61b00efcb0b8d5d3dadfac405e6af0b52f65a738732243872a494de007a7e0d` |

### Calibration artifact

| File | Source | License | SHA-256 |
|---|---|---|---|
| `calibration_results.json` | Output of `scripts/calibrate_weights.py` against this corpus (regenerable) | Apache-2.0 (273V) | `119a51d14e4f8435bc63619c45cf659a17feda80c7ac21e9a95152507b65e925` |
