import json
from pathlib import Path

import faiss

from rag.config import CHUNKS_PATH, INDEX_PATH
from rag.embeddings import load_embedding_model
from rag.retriever import retrieve_articles


def test_retrieval_returns_expected_article() -> None:
    model = load_embedding_model()
    index = faiss.read_index(str(INDEX_PATH))

    with Path(CHUNKS_PATH).open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    results = retrieve_articles(
        question="ما هو سن الرشد؟",
        model=model,
        index=index,
        chunks=chunks,
        top_k=5,
    )

    assert len(results) == 5
    assert results[0]["article_number"] == 44




import numpy as np
import pytest


def test_retrieval_rejects_mismatched_index_and_chunks() -> None:
    class FakeModel:
        def encode(self, questions, normalize_embeddings):
            return np.zeros((1, 2), dtype="float32")

    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0]], dtype="float32"))

    chunks = []

    with pytest.raises(ValueError, match="chunk collection"):
        retrieve_articles(
            question="What is the age of majority?",
            model=FakeModel(),
            index=index,
            chunks=chunks,
            top_k=5,
        )
