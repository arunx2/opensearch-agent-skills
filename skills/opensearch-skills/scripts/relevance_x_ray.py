#!/usr/bin/env python3
"""Evidence-backed relevance diagnostics for OpenSearch.

Commands:
    preflight-check
    inspect-index --index NAME
    explain --index NAME --query TEXT_OR_JSON --doc-id ID
    suggest-synonyms --index NAME --query-term TERM --doc-id ID
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.client import (  # noqa: E402
    build_client,
    can_connect,
    preflight_check_cluster,
    resolve_http_auth,
)
from lib.explain_parser import ExplainSummary, parse_explain, to_plain_english  # noqa: E402
from lib.relevance_diagnostics import (  # noqa: E402
    build_knn_parameter_sweep,
    compact_hit_context,
    find_rank,
    flatten_mapping,
    inspect_query,
    mapped_text_fields,
    parse_query_input,
)
from lib.report import build_diagnosis_report  # noqa: E402
from lib.rules_engine import evaluated_rule_names, run_all_rules  # noqa: E402
from lib.synonym_suggester import (  # noqa: E402
    analyze_source_document,
    fetch_sample_documents,
    mine_candidate_synonyms,
    validate_synonym_candidate,
)


def _preflight_result(args) -> dict:
    return preflight_check_cluster(
        auth_mode=getattr(args, "auth_mode", "") or "",
        username=getattr(args, "username", "") or "",
        password=getattr(args, "password", "") or "",
    )


def _checked_client(args):
    """Preflight in this process so diagnostic commands never bootstrap Docker."""
    result = _preflight_result(args)
    if result.get("status") != "available":
        raise RuntimeError(result.get("message") or "OpenSearch preflight failed.")
    http_auth = resolve_http_auth()
    for use_ssl in (True, False):
        client = build_client(use_ssl=use_ssl, http_auth=http_auth)
        connected, _ = can_connect(client)
        if connected:
            return client
    raise RuntimeError(
        "OpenSearch became unavailable after preflight; no diagnostic request was sent."
    )


def _index_context(client, index: str) -> tuple[dict, dict, dict]:
    mapping_response = client.indices.get_mapping(index=index)
    settings_response = client.indices.get_settings(index=index)
    if len(mapping_response) != 1:
        raise RuntimeError(
            f"Index expression '{index}' resolved to {len(mapping_response)} indices; "
            "provide one concrete index so scores and mappings are unambiguous."
        )
    concrete_index = next(iter(mapping_response))
    properties = (
        mapping_response[concrete_index].get("mappings", {}).get("properties", {})
    )
    return mapping_response, settings_response, properties


def _search(
    client,
    index: str,
    body: dict,
    top_k: int,
    search_pipeline: str = "",
    explain: bool = False,
) -> dict:
    request = copy.deepcopy(body)
    request["size"] = top_k
    request["from"] = 0
    request.pop("search_after", None)
    request["explain"] = explain
    params = {"search_pipeline": search_pipeline} if search_pipeline else None
    kwargs = {"index": index, "body": request}
    if params:
        kwargs["params"] = params
    return client.search(**kwargs)


def _hits(response: dict) -> list[dict]:
    return response.get("hits", {}).get("hits", []) or []


def _analyze(client, index: str, analyzer: str, text: str) -> list[str]:
    response = client.indices.analyze(
        index=index,
        body={"analyzer": analyzer, "text": text},
    )
    return [
        str(token.get("token", "")).lower()
        for token in response.get("tokens", [])
        if token.get("token")
    ]


def _build_analyzer_evidence(
    client,
    index: str,
    doc_id: str,
    metadata,
    mapping_properties: dict,
) -> tuple[dict, list[str]]:
    """Use configured analyzers and target term vectors; never infer by spelling."""
    flattened = flatten_mapping(mapping_properties)
    candidate_fields = sorted(
        field_name.split("^", 1)[0]
        for field_name in metadata.query_fields
        if field_name.split("^", 1)[0] in flattened
    )
    divergent_fields = [
        field_name
        for field_name in candidate_fields
        if flattened[field_name].get("search_analyzer")
        and flattened[field_name].get("search_analyzer")
        != flattened[field_name].get("analyzer", "standard")
    ]
    if not divergent_fields:
        return {}, []

    limitations: list[str] = []
    try:
        term_vectors = client.termvectors(
            index=index,
            id=doc_id,
            body={
                "fields": divergent_fields,
                "field_statistics": False,
                "term_statistics": False,
                "positions": False,
                "offsets": False,
                "payloads": False,
            },
        ).get("term_vectors", {})
    except Exception as exc:
        return {}, [f"Analyzer comparison skipped because term vectors failed: {exc}"]

    evidence: dict = {}
    for field_name in divergent_fields:
        spec = flattened[field_name]
        query_values = metadata.field_queries.get(field_name) or metadata.query_terms
        query_text = " ".join(query_values).strip()
        if not query_text:
            continue
        index_analyzer = spec.get("analyzer", "standard")
        search_analyzer = spec.get("search_analyzer", index_analyzer)
        try:
            evidence[field_name] = {
                "index_tokens": _analyze(client, index, index_analyzer, query_text),
                "search_tokens": _analyze(client, index, search_analyzer, query_text),
                "target_tokens": sorted(
                    (term_vectors.get(field_name, {}).get("terms") or {}).keys()
                ),
            }
        except Exception as exc:
            limitations.append(
                f"Analyzer comparison for field '{field_name}' failed: {exc}"
            )
    return evidence, limitations


def _explain_target(client, index: str, doc_id: str, query: dict) -> ExplainSummary:
    response = client.explain(index=index, id=doc_id, body={"query": query})
    return parse_explain(
        response.get("explanation") or {},
        doc_matched=response.get("matched"),
    )


def cmd_preflight_check(args) -> None:
    result = _preflight_result(args)
    print(json.dumps(result, indent=2))
    if result.get("status") != "available":
        raise RuntimeError(result.get("message") or "OpenSearch preflight failed.")


def cmd_inspect_index(args) -> None:
    client = _checked_client(args)
    mapping, settings, _ = _index_context(client, args.index)
    print(json.dumps({"mapping": mapping, "settings": settings}, indent=2))


def cmd_explain(args) -> None:
    client = _checked_client(args)
    mapping_response, _, mapping_properties = _index_context(client, args.index)
    search_body, plain_query = parse_query_input(args.query, mapping_properties)
    query = search_body["query"]
    metadata = inspect_query(query)
    limitations: list[str] = []

    try:
        search_response = _search(
            client,
            args.index,
            search_body,
            args.top_k,
            args.search_pipeline,
            explain=True,
        )
    except Exception as exc:
        limitations.append(f"Per-hit search explanations were unavailable: {exc}")
        search_response = _search(
            client,
            args.index,
            search_body,
            args.top_k,
            args.search_pipeline,
            explain=False,
        )

    hits = _hits(search_response)
    target_rank = find_rank(hits, args.doc_id)
    target_hit = next(
        (hit for hit in hits if str(hit.get("_id")) == str(args.doc_id)),
        None,
    )

    leg_summaries: dict[str, ExplainSummary] = {}
    if metadata.hybrid_legs:
        for index, leg_query in enumerate(metadata.hybrid_legs, start=1):
            try:
                leg_summaries[f"hybrid-leg-{index}"] = _explain_target(
                    client, args.index, args.doc_id, leg_query
                )
            except Exception as exc:
                limitations.append(f"Hybrid leg {index} could not be explained: {exc}")
        limitations.append(
            "Raw hybrid leg scores are not compared with pipeline weights because "
            "normalization is computed across the result set."
        )

    if target_hit and target_hit.get("_explanation"):
        summary = parse_explain(target_hit["_explanation"], doc_matched=True)
    elif not metadata.hybrid_legs:
        summary = _explain_target(client, args.index, args.doc_id, query)
    else:
        summary = ExplainSummary(
            total_score=float(target_hit.get("_score") or 0.0) if target_hit else 0.0,
            matched=bool(target_hit),
            match_known=bool(target_hit),
        )

    analyzer_evidence, analyzer_limitations = _build_analyzer_evidence(
        client,
        args.index,
        args.doc_id,
        metadata,
        mapping_properties,
    )
    limitations.extend(analyzer_limitations)

    knn_counterfactual = None
    if metadata.has_knn and not args.skip_knn_validation:
        swept_query, before_params, after_params = build_knn_parameter_sweep(query)
        if swept_query is None:
            limitations.append(
                "k-NN recall was not evaluated because the query has no explicit k "
                "or ef_search parameter to sweep."
            )
        else:
            swept_body = copy.deepcopy(search_body)
            swept_body["query"] = swept_query
            swept_hits = _hits(
                _search(
                    client,
                    args.index,
                    swept_body,
                    args.top_k,
                    args.search_pipeline,
                    explain=False,
                )
            )
            knn_counterfactual = {
                "before_rank": target_rank,
                "after_rank": find_rank(swept_hits, args.doc_id),
                "before_params": before_params,
                "after_params": after_params,
            }

    concrete_index = next(iter(mapping_response))
    mapping_fields = flatten_mapping(mapping_properties)
    mapping_fields.update(
        mapping_response[concrete_index].get("mappings", {}).get("runtime", {})
    )
    context = {
        "mapping_properties": mapping_fields,
        "filter_or_exact_fields": sorted(metadata.exact_fields),
        "referenced_fields": sorted(metadata.referenced_fields),
        "analysis_by_field": analyzer_evidence,
        "summary": summary,
        "query_terms": metadata.query_terms,
        "knn_counterfactual": knn_counterfactual,
    }
    findings = run_all_rules(context)
    evaluated_rules = evaluated_rule_names(context)
    hit_context = compact_hit_context(hits)
    for item, hit in zip(hit_context, hits):
        explanation = hit.get("_explanation")
        if explanation:
            lines = to_plain_english(parse_explain(explanation, doc_matched=True))
            if lines:
                item["score_evidence"] = lines[0]
    competitor_context = [
        item for item in hit_context if item["id"] != str(args.doc_id)
    ]

    if metadata.hybrid_legs:
        limitations.append(
            "Hybrid imbalance was not evaluated because normalized per-leg "
            "contributions are not exposed by this request."
        )
    if not metadata.query_terms and plain_query is None:
        limitations.append(
            "Vocabulary diagnostics were not evaluated because no textual query "
            "terms could be extracted from the DSL."
        )

    report = build_diagnosis_report(
        index=args.index,
        query_text=args.query,
        doc_id=args.doc_id,
        summary=summary,
        findings=findings,
        search_context={
            "target_rank": target_rank,
            "top_k": args.top_k,
            "top_hits": competitor_context,
        },
        evaluated_rules=evaluated_rules,
        limitations=limitations,
        leg_summaries=leg_summaries,
    )
    print(report)

    if args.raw:
        print("\nRAW SEARCH RESPONSE")
        print("-" * 72)
        print(json.dumps(search_response, indent=2))


def _synonym_search_fn(fields: list[str], top_k: int):
    def search_fn(client, index: str, query_text: str) -> list[str]:
        response = client.search(
            index=index,
            body={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": fields,
                        "operator": "or",
                    }
                },
            },
        )
        return [str(hit.get("_id")) for hit in _hits(response)]

    return search_fn


def cmd_suggest_synonyms(args) -> None:
    client = _checked_client(args)
    _, _, mapping_properties = _index_context(client, args.index)
    available_fields = mapped_text_fields(mapping_properties)
    fields = (
        [field.strip() for field in args.fields.split(",") if field.strip()]
        if args.fields
        else available_fields
    )
    unknown_fields = sorted(set(fields) - set(available_fields))
    if unknown_fields:
        raise ValueError(f"Synonym fields are not mapped text fields: {unknown_fields}")
    if not fields:
        raise ValueError("No text fields are available for synonym mining.")

    docs = fetch_sample_documents(
        client,
        args.index,
        fields,
        size=args.sample_size,
    )
    corpus_term_lists = [
        analyze_source_document(client, args.index, doc, fields) for doc in docs
    ]
    target_doc = client.get(index=args.index, id=args.doc_id)
    target_terms = analyze_source_document(
        client,
        args.index,
        target_doc.get("_source", {}),
        fields,
    )

    candidates = mine_candidate_synonyms(
        query_term=args.query_term,
        target_doc_terms=target_terms,
        corpus_term_lists=corpus_term_lists,
        min_support=args.min_support,
    )
    if not candidates:
        print(
            f"No corpus-supported candidate was found for '{args.query_term}' "
            f"in the {len(docs)} sampled documents."
        )
        return

    search_fn = _synonym_search_fn(fields, args.top_k)
    supported: list[tuple] = []
    rejected: list[tuple] = []
    for candidate in candidates:
        validation = validate_synonym_candidate(
            client,
            args.index,
            args.query_term,
            candidate,
            target_doc_id=str(args.doc_id),
            search_fn=search_fn,
        )
        item = (candidate, validation)
        (supported if validation.get("improved") else rejected).append(item)

    print(
        "Validation query shape: multi_match(operator=or), "
        f"fields={fields}, top_k={args.top_k}"
    )
    if supported:
        print(f"Supported candidates for '{args.query_term}':")
        for candidate, validation in supported:
            print(
                f"  - '{candidate.candidate}': support_docs={candidate.support}, "
                f"P(candidate|query)={candidate.confidence:.3f}, "
                f"jaccard={candidate.association:.3f}, "
                f"rank={validation['before_rank']}->{validation['after_rank']}"
            )
    else:
        print("No candidate improved the target rank; no synonym is recommended.")
    if rejected:
        print("Rejected after rank validation:")
        for candidate, validation in rejected:
            print(
                f"  - '{candidate.candidate}': "
                f"rank={validation['before_rank']}->{validation['after_rank']}"
            )


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auth-mode",
        choices=["none", "default", "custom"],
        default="",
        help="Authentication mode; omit to auto-detect.",
    )
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_preflight = sub.add_parser("preflight-check", help="Probe cluster connectivity")
    _add_connection_arguments(p_preflight)
    p_preflight.set_defaults(func=cmd_preflight_check)

    p_inspect = sub.add_parser(
        "inspect-index",
        help="Dump mapping/settings for one concrete index",
    )
    p_inspect.add_argument("--index", required=True)
    _add_connection_arguments(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect_index)

    p_explain = sub.add_parser(
        "explain",
        help="Run the actual search and explain one target document",
    )
    p_explain.add_argument("--index", required=True)
    p_explain.add_argument(
        "--query",
        required=True,
        help="Plain text, a query clause, or a complete JSON search body",
    )
    p_explain.add_argument("--doc-id", required=True)
    p_explain.add_argument("--top-k", type=int, default=10)
    p_explain.add_argument("--search-pipeline", default="")
    p_explain.add_argument("--skip-knn-validation", action="store_true")
    p_explain.add_argument(
        "--raw",
        action="store_true",
        help="Also print the raw search response",
    )
    _add_connection_arguments(p_explain)
    p_explain.set_defaults(func=cmd_explain)

    p_syn = sub.add_parser(
        "suggest-synonyms",
        help="Mine candidates and retain only measured rank improvements",
    )
    p_syn.add_argument("--index", required=True)
    p_syn.add_argument("--query-term", required=True)
    p_syn.add_argument("--doc-id", required=True)
    p_syn.add_argument(
        "--fields",
        default="",
        help="Comma-separated mapped text fields; defaults to all text fields.",
    )
    p_syn.add_argument("--sample-size", type=int, default=200)
    p_syn.add_argument("--min-support", type=int, default=2)
    p_syn.add_argument("--top-k", type=int, default=20)
    _add_connection_arguments(p_syn)
    p_syn.set_defaults(func=cmd_suggest_synonyms)

    args = parser.parse_args()
    if getattr(args, "top_k", 1) < 1:
        parser.error("--top-k must be at least 1")
    if getattr(args, "sample_size", 1) < 1:
        parser.error("--sample-size must be at least 1")
    if getattr(args, "min_support", 1) < 1:
        parser.error("--min-support must be at least 1")
    try:
        args.func(args)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
