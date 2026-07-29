# Relevance X-Ray

**OpenSearch Agent Skills Hackathon 2026 submission.** Diagnoses *why* a
specific search result ranked where it did in OpenSearch by collecting
rank, competitor, analyzer, mapping, and explain evidence. It proposes a
fix only when that evidence supports one.

Submission: https://github.com/opensearch-project/opensearch-agent-skills/issues/86

## Problem

`opensearch-launchpad`'s evaluation guide already answers "is my index
good, on average, across many test queries" (nDCG, P@k, MRR). But when a
user has one specific query and one specific document they're confused
about — "why didn't 'wireless charger' return this product in the top 5?"
— the raw `_explain` tree alone does not establish a ranking cause. This
skill combines it with rank, competitor, mapping, and analyzer evidence and
explicitly abstains when that evidence is insufficient.

## What it does

Given an index, a query, and a document, Relevance X-Ray:

1. Fetches the index's mapping/analyzer configuration for context.
2. Runs the actual top-k search and parses the target's explain tree while
   preserving sum/max/product operations and separating score clauses from
   non-additive factors.
3. Runs a small rules engine against known anti-patterns: missing
   `.keyword` sub-fields, unavailable doc-value scoring fields, analyzer
   mismatches backed by analyzed tokens/term vectors, measured k-NN recall
   changes, and normalized hybrid output when available.
4. Optionally mines target-present synonym candidates with document-level
   support and retains only candidates that improve rank in a controlled
   query-expansion rerun.

Findings reuse the `[INDEX_MAPPING]` / `[MODEL_SELECTION]` /
`[SEARCH_PIPELINE]` / `[QUERY_TUNING]` tag vocabulary already established
by `opensearch-launchpad`'s evaluator, so the two skills read consistently.

## Try it

```bash
# 1. Start a local cluster (shared script, one directory up)
bash ../../scripts/start_opensearch.sh

# 2. Load the demo product-catalog index (has a deliberate vocabulary gap)
bash examples/demo_index_setup.sh

# 3. Diagnose why doc 1 ("Lightweight Running Trainers") doesn't show up
#    for a "sneakers" query
uv run python ../../scripts/relevance_x_ray.py explain \
  --index relevance-x-ray-demo --query sneakers --doc-id 1

# 4. Test corpus-supported candidates with a measured target-rank delta
uv run python ../../scripts/relevance_x_ray.py suggest-synonyms \
  --index relevance-x-ray-demo --query-term sneakers --doc-id 1
```

## Files

```
relevance-x-ray/
  SKILL.md                     Skill manifest (frontmatter + workflow)
  README.md                    This file
  examples/
    demo_index_setup.sh        Seeds a small sample index for demos

../../scripts/
  relevance_x_ray.py           CLI entrypoint (preflight-check, inspect-index,
                                explain, suggest-synonyms)
  lib/
    explain_parser.py          Preserves explain operators, clauses, and factors
    relevance_diagnostics.py   Inspects query DSL and builds counterfactuals
    rules_engine.py            Anti-pattern detection rules
    synonym_suggester.py       Candidate synonym mining + validation
    report.py                  Formats findings into the fixed diagnosis schema
    client.py                  (existing, shared) connection handling — reused as-is
```

## Testing

Follows the repo's convention: pure functions unit-tested with fixture
data, no live cluster required.

```bash
uv run pytest tests/test_agent_skills_explain_parser.py \
              tests/test_agent_skills_relevance_diagnostics.py \
              tests/test_agent_skills_relevance_x_ray.py \
              tests/test_agent_skills_rules_engine.py \
              tests/test_agent_skills_synonym_suggester.py \
              tests/test_agent_skills_report.py -v
```

The thin client-calling functions in `synonym_suggester.py`
(`fetch_sample_documents`, `simulate_synonym_analyzer`,
`validate_synonym_candidate`) are tested against a fake client object, not
a real cluster, matching the pattern used elsewhere in this repo.

## Design notes / scope for the hackathon build

- Built as an addition to a fork of this repo (not a standalone repo) so a
  PR back upstream is close to a diff rather than a rewrite, and so it
  inherits the existing `client.py` connection/auth primitives without
  allowing diagnostic commands to auto-bootstrap Docker.
- Vendor-neutral: pure OpenSearch REST API calls (`_search`, `_explain`,
  `_analyze`, `_termvectors`, `_mapping`), no proprietary dependencies.
  Supports endpoints and authentication modes handled by the shared client
  primitives.
- The synonym miner is a lightweight, dependency-free co-occurrence
  heuristic, not an embedding similarity search — this keeps it fully
  unit-testable with no extra ML dependency and no network calls in tests.
  A future iteration could swap in the vector engine for candidate mining.
- Scope was deliberately staged: BM25 explain parsing and the anti-pattern
  rules are the baseline-submittable core; hybrid/k-NN leg-splitting and
  the synonym suggester are the stretch layers described in the hackathon
  submission.
