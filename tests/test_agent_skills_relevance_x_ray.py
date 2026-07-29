"""CLI orchestration tests for Relevance X-Ray."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "skills" / "opensearch-skills" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import relevance_x_ray


def _term_explanation(doc_number=0, value=1.0):
    return {
        "value": value,
        "description": "sum of:",
        "details": [{
            "value": value,
            "description": (
                f"weight(title:wireless in {doc_number}) "
                "[PerFieldSimilarity], result of:"
            ),
            "details": [],
        }],
    }


class _Indices:
    def get_mapping(self, index):
        return {
            index: {
                "mappings": {
                    "properties": {
                        "title": {"type": "text"},
                        "description": {"type": "text"},
                        "price": {"type": "float"},
                    }
                }
            }
        }

    def get_settings(self, index):
        return {index: {"settings": {"index": {}}}}

    def analyze(self, index, body):
        text = body["text"]
        if isinstance(text, list):
            text = " ".join(text)
        return {
            "tokens": [
                {"token": token.strip(".,").lower()}
                for token in text.split()
                if token.strip(".,")
            ]
        }


class _ExplainClient:
    def __init__(self):
        self.indices = _Indices()
        self.search_requests = []

    def search(self, **kwargs):
        self.search_requests.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "competitor",
                        "_score": 2.0,
                        "_explanation": _term_explanation(0, 2.0),
                    },
                    {
                        "_id": "target",
                        "_score": 1.0,
                        "_explanation": _term_explanation(1, 1.0),
                    },
                ]
            }
        }


def test_checked_client_stops_after_failed_preflight(monkeypatch):
    monkeypatch.setattr(
        relevance_x_ray,
        "_preflight_result",
        lambda args: {"status": "no_cluster", "message": "unreachable"},
    )
    monkeypatch.setattr(
        relevance_x_ray,
        "build_client",
        lambda **kwargs: pytest.fail("no client must be built after failed preflight"),
    )
    with pytest.raises(RuntimeError, match="unreachable"):
        relevance_x_ray._checked_client(SimpleNamespace())


def test_checked_client_never_bootstraps_if_cluster_disappears(monkeypatch):
    monkeypatch.setattr(
        relevance_x_ray,
        "_preflight_result",
        lambda args: {"status": "available"},
    )
    monkeypatch.setattr(relevance_x_ray, "resolve_http_auth", lambda: None)
    monkeypatch.setattr(
        relevance_x_ray,
        "build_client",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        relevance_x_ray,
        "can_connect",
        lambda client: (False, False),
    )
    with pytest.raises(RuntimeError, match="became unavailable"):
        relevance_x_ray._checked_client(SimpleNamespace())


def test_explain_runs_actual_search_and_reports_target_rank(monkeypatch, capsys):
    client = _ExplainClient()
    monkeypatch.setattr(relevance_x_ray, "_checked_client", lambda args: client)
    args = SimpleNamespace(
        index="products",
        query="wireless",
        doc_id="target",
        top_k=10,
        search_pipeline="",
        skip_knn_validation=False,
        raw=False,
    )

    relevance_x_ray.cmd_explain(args)

    output = capsys.readouterr().out
    assert "Observed target rank: 2" in output
    assert "Competing hit rank 1: doc 'competitor'" in output
    assert "Competing hit rank 2: doc 'target'" not in output
    assert "Evidence: Matched term 'wireless'" in output
    assert "No supported root cause was established" in output
    assert "scoring behaved as expected" not in output
    assert client.search_requests[0]["body"]["explain"] is True


class _SynonymClient:
    def __init__(self):
        self.indices = _Indices()

    def search(self, index, body):
        if body["query"] == {"match_all": {}}:
            return {
                "hits": {
                    "hits": [
                        {
                            "_source": {
                                "title": "Sneakers also called trainers",
                                "description": "Footwear glossary",
                            }
                        },
                        {
                            "_source": {
                                "title": "Sneakers and trainers",
                                "description": "Shoe terminology",
                            }
                        },
                    ]
                }
            }
        query_text = body["query"]["multi_match"]["query"]
        ids = ["2"] if query_text == "sneakers" else ["1", "2"]
        return {"hits": {"hits": [{"_id": doc_id} for doc_id in ids]}}

    def get(self, index, id):
        return {
            "_source": {
                "title": "Lightweight running trainers",
                "description": "Daily trail shoe",
            }
        }


def test_suggest_synonyms_only_recommends_rank_validated_candidate(
    monkeypatch, capsys
):
    client = _SynonymClient()
    monkeypatch.setattr(relevance_x_ray, "_checked_client", lambda args: client)
    args = SimpleNamespace(
        index="products",
        query_term="sneakers",
        doc_id="1",
        fields="title,description",
        sample_size=20,
        min_support=2,
        top_k=20,
    )

    relevance_x_ray.cmd_suggest_synonyms(args)

    output = capsys.readouterr().out
    assert "Supported candidates" in output
    assert "'trainers'" in output
    assert "rank=None->1" in output
