"""Deterministic helpers for evidence-backed relevance diagnostics."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field


_TEXT_FIELD_TYPES = {"text", "match_only_text", "search_as_you_type"}
_SCRIPT_FIELD_RE = re.compile(r"""doc\s*\[\s*['"]([^'"]+)['"]\s*\]""")


@dataclass
class QueryMetadata:
    query_terms: list[str] = field(default_factory=list)
    query_fields: set[str] = field(default_factory=set)
    exact_fields: set[str] = field(default_factory=set)
    referenced_fields: set[str] = field(default_factory=set)
    field_queries: dict[str, list[str]] = field(default_factory=dict)
    hybrid_legs: list[dict] = field(default_factory=list)
    has_knn: bool = False


def flatten_mapping(properties: dict, prefix: str = "") -> dict[str, dict]:
    """Flatten object properties and multi-fields to dotted field names."""
    flattened: dict[str, dict] = {}
    for name, spec in (properties or {}).items():
        if not isinstance(spec, dict):
            continue
        path = f"{prefix}.{name}" if prefix else name
        flattened[path] = spec
        if isinstance(spec.get("properties"), dict):
            flattened.update(flatten_mapping(spec["properties"], path))
        for sub_name, sub_spec in (spec.get("fields") or {}).items():
            if isinstance(sub_spec, dict):
                flattened[f"{path}.{sub_name}"] = sub_spec
    return flattened


def mapped_text_fields(mapping_properties: dict) -> list[str]:
    flattened = flatten_mapping(mapping_properties)
    return sorted(
        field_name
        for field_name, spec in flattened.items()
        if spec.get("type", "object") in _TEXT_FIELD_TYPES
    )


def parse_query_input(query_text: str, mapping_properties: dict) -> tuple[dict, str | None]:
    """Return a complete search body and the plain-text query, when applicable."""
    try:
        parsed = json.loads(query_text)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if parsed is not None:
        if not isinstance(parsed, dict):
            raise ValueError("JSON query input must be an object.")
        if "query" in parsed:
            if not isinstance(parsed["query"], dict):
                raise ValueError("The JSON body's 'query' value must be an object.")
            return copy.deepcopy(parsed), None
        return {"query": copy.deepcopy(parsed)}, None

    fields = mapped_text_fields(mapping_properties)
    if not fields:
        raise ValueError("The index has no mapped text fields for a plain-text query.")
    return {
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": fields,
            }
        }
    }, query_text


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def inspect_query(query: dict) -> QueryMetadata:
    """Extract evidence inputs from common OpenSearch query DSL clauses."""
    metadata = QueryMetadata()

    def walk(node, in_filter: bool = False) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, in_filter=in_filter)
            return
        if not isinstance(node, dict):
            return

        for clause, value in node.items():
            if clause == "hybrid" and isinstance(value, dict):
                legs = value.get("queries") or []
                metadata.hybrid_legs.extend(
                    copy.deepcopy(leg) for leg in legs if isinstance(leg, dict)
                )
            if clause in {"knn", "neural"}:
                metadata.has_knn = True
            if clause in {"filter", "must_not"}:
                walk(value, in_filter=True)
                continue

            if clause in {"match", "match_phrase", "term", "terms", "prefix", "wildcard"}:
                if isinstance(value, dict):
                    for field_name, field_value in value.items():
                        metadata.query_fields.add(field_name)
                        if clause in {"term", "terms"} or in_filter:
                            metadata.exact_fields.add(field_name)
                        if isinstance(field_value, dict):
                            field_terms = _strings(field_value.get("query"))
                            field_terms.extend(_strings(field_value.get("value")))
                        else:
                            field_terms = _strings(field_value)
                        metadata.query_terms.extend(field_terms)
                        metadata.field_queries.setdefault(field_name, []).extend(field_terms)
            elif clause == "multi_match" and isinstance(value, dict):
                field_terms = _strings(value.get("query"))
                fields = _strings(value.get("fields"))
                metadata.query_terms.extend(field_terms)
                metadata.query_fields.update(fields)
                for field_name in fields:
                    metadata.field_queries.setdefault(field_name.split("^", 1)[0], []).extend(
                        field_terms
                    )
            elif clause in {"query_string", "simple_query_string"} and isinstance(value, dict):
                metadata.query_terms.extend(_strings(value.get("query")))
                metadata.query_fields.update(_strings(value.get("fields")))
            elif clause == "field_value_factor" and isinstance(value, dict):
                field_name = value.get("field")
                if isinstance(field_name, str):
                    metadata.referenced_fields.add(field_name)
            elif clause == "script" and isinstance(value, dict):
                source = value.get("source")
                if isinstance(source, str):
                    metadata.referenced_fields.update(_SCRIPT_FIELD_RE.findall(source))

            walk(value, in_filter=in_filter)

    walk(query)
    metadata.query_terms = list(dict.fromkeys(metadata.query_terms))
    return metadata


def build_knn_parameter_sweep(query: dict) -> tuple[dict | None, dict, dict]:
    """Increase k/ef_search in a copied query for a controlled recall check."""
    swept = copy.deepcopy(query)
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    changed = False

    def walk(node) -> None:
        nonlocal changed
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        for clause, value in node.items():
            if clause in {"knn", "neural"} and isinstance(value, dict):
                for field_name, spec in value.items():
                    if not isinstance(spec, dict):
                        continue
                    current_k = spec.get("k")
                    if isinstance(current_k, int) and current_k > 0:
                        next_k = max(current_k * 2, 20)
                        before[f"{field_name}.k"] = current_k
                        after[f"{field_name}.k"] = next_k
                        spec["k"] = next_k
                        changed = True
                    method = spec.get("method_parameters")
                    if isinstance(method, dict):
                        current_ef = method.get("ef_search")
                        if isinstance(current_ef, int) and current_ef > 0:
                            next_ef = max(current_ef * 2, 100)
                            before[f"{field_name}.ef_search"] = current_ef
                            after[f"{field_name}.ef_search"] = next_ef
                            method["ef_search"] = next_ef
                            changed = True
            walk(value)

    walk(swept)
    return (swept if changed else None), before, after


def find_rank(hits: list[dict], doc_id: str) -> int | None:
    target = str(doc_id)
    for rank, hit in enumerate(hits, start=1):
        if str(hit.get("_id")) == target:
            return rank
    return None


def compact_hit_context(hits: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "id": str(hit.get("_id")),
            "score": hit.get("_score"),
        }
        for rank, hit in enumerate(hits, start=1)
    ]
