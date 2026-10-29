from typing import Dict
from pydantic import BaseModel
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # app_config -> project_code -> repo root

class TestConfig(BaseModel):
    run_in_test_mode: bool

class LLM(BaseModel):
    api_key: str
    base_url: str
    llm_model: str

class Logging_and_records(BaseModel):
    output_path: str

class DataBase(BaseModel):
    manual_pdf_path: str
    pdf_reader: str = "pypdf"  # which library to use to read the PDF: "pypdf" or "docling"

class RAG(BaseModel):
    embedding_model: str
    top_k: int
    retrieval_mode: str = "flat"
    flat_chunk_size: int = 1000
    flat_chunk_overlap: int = 200
    parent_chunk_size: int = 3000
    parent_chunk_overlap: int = 300
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    search_mode: str = "semantic"   # "semantic" or "hybrid"
    bm25_top_k: int = 20
    rrf_k: int = 60
    build_faiss_db: bool = True
    faiss_db_dir: str = "outputs/faiss_db"

class STTConfig(BaseModel):
    model_size: str
    num_beams: int
    confidence_threshold: float
    compression_ratio_threshold: float

class QACacheConfig(BaseModel):
    enabled: bool
    cache_dir: str
    similarity_threshold: float

class Settings(BaseModel):
    llm: LLM
    logging_and_records: Logging_and_records
    database: DataBase
    rag: RAG
    qa_cache: QACacheConfig




def load_config(path: str = "app_config/conf.yaml") -> Settings:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    settings = Settings(**data)
    # Anchor relative paths to the repo root so they work regardless of the
    # process's current working directory (important for ADK/agent runners).
    settings.database.manual_pdf_path = str(REPO_ROOT / settings.database.manual_pdf_path)
    settings.qa_cache.cache_dir       = str(REPO_ROOT / settings.qa_cache.cache_dir)
    settings.rag.faiss_db_dir         = str(REPO_ROOT / settings.rag.faiss_db_dir)
    return settings

def _flatten_for_log(data: dict, indent: int = 0) -> list[str]:
    lines = []
    pad = "  " * indent
    for k, v in (data or {}).items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.extend(_flatten_for_log(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{pad}{k}:")
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    lines.append(f"{pad}  [{i}]:")
                    lines.extend(_flatten_for_log(item, indent + 2))
                else:
                    lines.append(f"{pad}  [{i}] {item}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return lines

def log_app_settings():
    settings_dict = app_settings.model_dump()  # pydantic v2 (use .dict() if pydantic v1)
    from project_code.utils.utils import log_boxed
    log_boxed("App Settings", _flatten_for_log(settings_dict))

app_settings = load_config()
