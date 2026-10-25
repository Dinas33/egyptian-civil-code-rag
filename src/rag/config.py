
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen2.5:3b"

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "civil_code.faiss"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "rag_chunks.json"

TOP_K = 5
EMBEDDING_DIMENSION = 1024

MLFLOW_EXPERIMENT = "egyptian-civil-code-rag"
