
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve_articles(
    question: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS index contains {index.ntotal} vectors, "
            f"but the chunk collection contains {len(chunks)} items."
        )

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append(
            {
                "score": float(score),
                "article_number": chunk["article_number"],
                "id": chunk["id"],
                "text_ar": chunk["text_ar"],
                "text_en": chunk["text_en"],
            }
        )

    return results
