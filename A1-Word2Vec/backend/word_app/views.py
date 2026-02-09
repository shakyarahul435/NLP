from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import os
import json
import math
from django.conf import settings

# Paths to your local skip-gram negative sampling outputs
EMBEDDINGS_DIR = getattr(
    settings,
    "SKIPGRAM_NEG_DIR",
    os.path.join(settings.BASE_DIR, "output"),
)
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "skipgram_neg_embeddings.json")
CORPUS_PATH = os.path.join(EMBEDDINGS_DIR, "corpus.json")

# Caches
_embeddings = None         # dict: word -> vector (list[float])
_norms = None              # dict: word -> float
_vocab = None              # set[str]
_corpus = None             # list[str] or other structure from CORPUS_PATH


def _load_skipgram_neg():
    """
    Lazy-load embeddings and corpus from the local JSON files.
    Supports formats:
      - {"word": [...], ...}
      - {"embeddings": {"word": [...], ...}}
      - [{"word": "w", "vector": [...]}]
    """
    global _embeddings, _norms, _vocab, _corpus
    if _embeddings is not None:
        return

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw = data.get("embeddings", data)
    elif isinstance(data, list):
        raw = {item["word"]: item.get("vector") or item.get("embedding") for item in data}
    else:
        raise ValueError("Unsupported embeddings JSON format.")

    # Build lowercase vocab and precompute norms
    _embeddings = {}
    _norms = {}
    for w, vec in raw.items():
        if not isinstance(vec, (list, tuple)):
            continue
        v = [float(x) for x in vec]
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            continue
        wl = w.lower()
        _embeddings[wl] = v
        _norms[wl] = norm

    # Optional: use corpus to refine/validate vocab and cache corpus for response
    _corpus = None
    if os.path.exists(CORPUS_PATH):
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            # If corpus is a list of tokens, keep intersection and cache corpus
            if isinstance(corpus, list) and all(isinstance(t, str) for t in corpus):
                _corpus = [t.lower() for t in corpus]
                corpus_set = set(_corpus)
                _embeddings = {w: v for w, v in _embeddings.items() if w in corpus_set}
                _norms = {w: _norms[w] for w in _embeddings.keys()}
            else:
                # Cache as-is for transparency even if not token list
                _corpus = corpus
        except Exception:
            # If corpus parsing fails, continue with embeddings-only vocab
            _corpus = None

    _vocab = set(_embeddings.keys())


def _most_similar(word: str, topn: int = 5):
    w = word.lower()
    if w not in _embeddings:
        return None

    v1 = _embeddings[w]
    n1 = _norms[w]
    scores = []

    # Assume all vectors share the same dimensionality
    for other, v2 in _embeddings.items():
        if other == w:
            continue
        n2 = _norms[other]
        dot = sum(a * b for a, b in zip(v1, v2))
        sim = dot / (n1 * n2)
        scores.append((other, sim))

    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:topn]


class SimilarityView(APIView):
    def post(self, request):
        word = (request.data.get("word") or "").strip().lower()
        topn = int(request.data.get("topn", 5))

        try:
            _load_skipgram_neg()

            if not word:
                return Response({"error": "Missing 'word' in request"}, status=status.HTTP_400_BAD_REQUEST)

            if word not in _vocab:
                # Still return corpus (if available) along with the error
                return Response(
                    {"error": "Word not in vocabulary", "corpus": _corpus or []},
                    status=status.HTTP_404_NOT_FOUND,
                )

            result = _most_similar(word, topn=topn)
            if result is None:
                return Response(
                    {"error": "Word not in vocabulary", "corpus": _corpus or []},
                    status=status.HTTP_404_NOT_FOUND,
                )

            data = [{"word": w, "score": round(s, 4)} for w, s in result]
            return Response(
                {
                    "word": word,
                    "results": data,
                    "corpus": _corpus or [],
                    "vocab_size": len(_vocab),
                },
                status=status.HTTP_200_OK,
            )

        except FileNotFoundError as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)