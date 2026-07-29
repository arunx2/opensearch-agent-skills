#!/bin/bash
# Sets up a small sample product-catalog index for demoing Relevance X-Ray.
#
# Deliberately includes a vocabulary gap ("sneakers" vs "trainers") and a
# missing-.keyword-subfield anti-pattern, so the demo has something real
# to diagnose.
#
# Usage:
#   bash examples/demo_index_setup.sh [host] [port]
#
# Prerequisites: an OpenSearch cluster reachable at host:port (default
# localhost:9200). Start one with:
#   bash ../../scripts/start_opensearch.sh

set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-9200}"
BASE="http://${HOST}:${PORT}"
INDEX="relevance-x-ray-demo"

echo "Creating index '${INDEX}' at ${BASE}..." >&2

curl -s -X DELETE "${BASE}/${INDEX}" >/dev/null 2>&1 || true

curl -s -X PUT "${BASE}/${INDEX}" -H 'Content-Type: application/json' -d '{
  "mappings": {
    "properties": {
      "title": {"type": "text"},
      "description": {"type": "text"},
      "brand": {"type": "text"},
      "category": {"type": "keyword"},
      "price": {"type": "float"}
    }
  }
}' > /dev/null

echo "Indexing sample documents..." >&2

# Doc 1: uses "trainers" -- will NOT match a "sneakers" query verbatim,
# demonstrating the vocabulary_mismatch rule.
curl -s -X PUT "${BASE}/${INDEX}/_doc/1" -H 'Content-Type: application/json' -d '{
  "title": "Lightweight Running Trainers",
  "description": "Breathable mesh trainers built for daily runs and light trail use.",
  "brand": "TrailCo",
  "category": "footwear",
  "price": 79.99
}' > /dev/null

# Doc 2: uses "sneakers" directly -- will match a "sneakers" query.
curl -s -X PUT "${BASE}/${INDEX}/_doc/2" -H 'Content-Type: application/json' -d '{
  "title": "Classic Canvas Sneakers",
  "description": "Everyday canvas sneakers with a rubber sole.",
  "brand": "UrbanStep",
  "category": "footwear",
  "price": 49.99
}' > /dev/null

# Doc 3: another "trainers" product.
curl -s -X PUT "${BASE}/${INDEX}/_doc/3" -H 'Content-Type: application/json' -d '{
  "title": "Trail Trainers for Wet Weather",
  "description": "Waterproof trainers with reinforced toe cap.",
  "brand": "TrailCo",
  "category": "footwear",
  "price": 89.99
}' > /dev/null

# Doc 4: unrelated product, for contrast.
curl -s -X PUT "${BASE}/${INDEX}/_doc/4" -H 'Content-Type: application/json' -d '{
  "title": "Wireless Charging Pad",
  "description": "10W fast wireless charger compatible with most phone cases.",
  "brand": "VoltEdge",
  "category": "electronics",
  "price": 24.99
}' > /dev/null

# Docs 5-6: independent bridge documents provide document-level evidence that
# "sneakers" and "trainers" are used for the same product category. The
# suggester requires at least two supporting documents by default.
curl -s -X PUT "${BASE}/${INDEX}/_doc/5" -H 'Content-Type: application/json' -d '{
  "title": "Everyday Sneakers and Trainers",
  "description": "A casual footwear listing using both sneakers and trainers terminology.",
  "brand": "MetroSole",
  "category": "footwear",
  "price": 59.99
}' > /dev/null

curl -s -X PUT "${BASE}/${INDEX}/_doc/6" -H 'Content-Type: application/json' -d '{
  "title": "Road Sneakers, also known as Trainers",
  "description": "Light road shoes described as sneakers and trainers.",
  "brand": "PaceWorks",
  "category": "footwear",
  "price": 69.99
}' > /dev/null

curl -s -X POST "${BASE}/${INDEX}/_refresh" > /dev/null

echo "Done. Try:" >&2
echo "  uv run python ../../scripts/relevance_x_ray.py explain \\
    --index ${INDEX} --query sneakers --doc-id 1 --exact-fields brand" >&2
echo "  uv run python ../../scripts/relevance_x_ray.py suggest-synonyms \\
    --index ${INDEX} --query-term sneakers --doc-id 1" >&2
