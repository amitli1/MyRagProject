import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI
import re

from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
import time

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"



# Lazily-initialized, process-wide singleton so the (relatively heavy)
# sentence-transformers model is only loaded once.
_embedding_model = None


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())



def _get_embedding_model(model_name: str) -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logging.info("Loading embedding model '%s'", model_name)
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


@dataclass
class Chunk:
    """A single chunk of document text, plus debug-only provenance info
    (page number / section header) so we can trace retrieved context back
    to its source location in the manual. Not shown to the end user."""
    text: str
    page: int = -1
    header: str = ""
    parent_id: int = -1  # index into ManualIndex.parents; -1 if this chunk has no parent (flat mode)


@dataclass
class ManualIndex:
    chunks: List[Chunk] = field(default_factory=list)
    parents: List[Chunk] = field(default_factory=list)
    index: "faiss.IndexFlatIP" = None
    bm25: "BM25Okapi" = None          # NEW
    embedding_model: str = EMBEDDING_MODEL_NAME
    retrieval_mode: str = "flat"

# Matches lines that look like section/chapter headers in the extracted PDF text,
# e.g. "CHAPTER 3", "3.2 DRIVER CONTROLS", "SECTION 4 - MAINTENANCE".
_HEADER_RE = re.compile(
    r"^\s*((?:CHAPTER|SECTION)\s+[\dIVXLC]+.*|\d+(?:\.\d+)*\s+[A-Z][A-Z0-9 ,/&\-]{3,})\s*$"
)

def _build_bm25_index(chunks: List[Chunk]):
    if not chunks:
        return None
    tokenized = [_tokenize(c.text) for c in chunks]
    return BM25Okapi(tokenized)

def _extract_pages_from_pdf_pypdf(document_path):
    """Extract raw text from every page of the PDF using 'pypdf', keeping page numbers (1-based)."""
    reader = PdfReader(document_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text))
    return pages


def _extract_pages_from_pdf_docling(document_path):
    """Extract raw text from every page of the PDF using 'docling', keeping page numbers (1-based).

    Docling performs layout-aware parsing (tables, reading order, etc.) which can
    yield cleaner text than a plain text-extraction library, at the cost of being
    slower and requiring the optional 'docling' dependency.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise ImportError(
            "pdf_reader='docling' requires the 'docling' package. "
            "Install it with: pip install docling"
        ) from e

    logging.info("Extracting PDF text with docling: '%s'", document_path)
    converter = DocumentConverter()
    result = converter.convert(str(document_path))
    doc = result.document

    pages_text: dict = {}
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        prov = getattr(item, "prov", None)
        page_number = prov[0].page_no if prov else 1
        pages_text.setdefault(page_number, []).append(text)

    return [(page_number, "\n".join(texts)) for page_number, texts in sorted(pages_text.items())]


def _extract_pages_from_pdf(document_path, pdf_reader: str = "pypdf"):
    """Extract raw text from every page of the PDF, keeping page numbers (1-based).

    Dispatches to the configured backend library, chosen via database.pdf_reader
    in conf.yaml: "pypdf" (fast, default) or "docling" (layout-aware parsing).
    """
    pdf_reader = (pdf_reader or "pypdf").lower()
    if pdf_reader == "docling":
        return _extract_pages_from_pdf_docling(document_path)
    if pdf_reader == "pypdf":
        return _extract_pages_from_pdf_pypdf(document_path)
    raise ValueError(f"Unsupported pdf_reader '{pdf_reader}'. Expected 'pypdf' or 'docling'.")


def _extract_text_from_pdf(document_path, pdf_reader: str = "pypdf"):
    """Extract raw text from every page of the PDF (kept for backward compatibility)."""
    pages = _extract_pages_from_pdf(document_path, pdf_reader=pdf_reader)
    return "\n".join(text for _, text in pages)


def _find_last_header(text, default=""):
    """Best-effort detection of the most recent section/chapter header within a
    block of raw (pre-whitespace-collapsed) text. Debug-only heuristic."""
    last_header = default
    for line in text.splitlines():
        match = _HEADER_RE.match(line)
        if match:
            last_header = match.group(1).strip()
    return last_header


def _chunk_text(text, chunk_size, overlap):
    """Split text into overlapping chunks so context fits the model window.
    Kept for backward compatibility; returns plain strings with no provenance."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - overlap
    return chunks

def _chunk_pages_parent_child(pages, parent_size, parent_overlap, child_size, child_overlap):
    parents: List[Chunk] = []
    children: List[Chunk] = []
    current_header = ""
    for page_number, raw_text in pages:
        current_header = _find_last_header(raw_text, default=current_header)
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            continue

        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + parent_size, text_len)
            parent_text = text[start:end]
            parent_id = len(parents)
            parents.append(Chunk(text=parent_text, page=page_number, header=current_header))

            # split this parent into smaller children for embedding/search
            c_start = 0
            p_len = len(parent_text)
            while c_start < p_len:
                c_end = min(c_start + child_size, p_len)
                children.append(Chunk(
                    text=parent_text[c_start:c_end],
                    page=page_number,
                    header=current_header,
                    parent_id=parent_id,
                ))
                if c_end == p_len:
                    break
                c_start = c_end - child_overlap

            if end == text_len:
                break
            start = end - parent_overlap
    return parents, children


def _embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Embed a list of texts with the sentence-transformers model, L2-normalized
    so that inner-product search in FAISS is equivalent to cosine similarity."""
    model = _get_embedding_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def _build_faiss_index(chunks: List[Chunk], model_name: str = EMBEDDING_MODEL_NAME) -> "faiss.IndexFlatIP":
    """Build a FAISS inner-product (cosine) index over the embedded chunks."""
    embeddings = _embed_texts([c.text for c in chunks], model_name)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def _faiss_db_paths(faiss_db_dir: str):
    """Returns the (index_file, metadata_file) paths used to persist a ManualIndex."""
    db_dir = Path(faiss_db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "faiss.index", db_dir / "meta.pkl"


def save_manual_index(manual_index: ManualIndex, faiss_db_dir: str) -> None:
    """Persist a ManualIndex (FAISS index + chunks/parents/bm25/metadata) to disk."""
    index_path, meta_path = _faiss_db_paths(faiss_db_dir)

    if manual_index.index is not None:
        faiss.write_index(manual_index.index, str(index_path))

    meta = {
        "chunks": manual_index.chunks,
        "parents": manual_index.parents,
        "bm25": manual_index.bm25,
        "embedding_model": manual_index.embedding_model,
        "retrieval_mode": manual_index.retrieval_mode,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    logging.info("Saved FAISS DB (%d chunks) to '%s'", len(manual_index.chunks), faiss_db_dir)


def load_manual_index(faiss_db_dir: str) -> ManualIndex:
    """Load a previously saved ManualIndex (FAISS index + metadata) from disk."""
    index_path, meta_path = _faiss_db_paths(faiss_db_dir)

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No saved FAISS DB found at '{faiss_db_dir}'. "
            f"Set rag.build_faiss_db=true in conf.yaml to build and save it first."
        )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    index = faiss.read_index(str(index_path))

    logging.info("Loaded FAISS DB (%d chunks) from '%s'", len(meta["chunks"]), faiss_db_dir)

    return ManualIndex(
        chunks=meta["chunks"],
        parents=meta["parents"],
        index=index,
        bm25=meta["bm25"],
        embedding_model=meta["embedding_model"],
        retrieval_mode=meta["retrieval_mode"],
    )


def _rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk indices.
    Each list is ordered best-first. Returns fused indices, best-first."""
    scores = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _retrieve_relevant_chunks(manual_index: ManualIndex, user_question, top_k,
                               search_mode: str = "semantic", bm25_top_k: int = 20,
                               rrf_k: int = 60) -> List[Chunk]:
    if not manual_index.chunks:
        return []

    question_vector = _embed_texts([user_question], manual_index.embedding_model)
    faiss_k = min(max(top_k, bm25_top_k) if search_mode == "hybrid" else top_k, len(manual_index.chunks))
    _, faiss_indices = manual_index.index.search(question_vector, faiss_k)
    faiss_rank = [i for i in faiss_indices[0] if i != -1]

    if search_mode == "hybrid" and manual_index.bm25 is not None:
        bm25_scores = manual_index.bm25.get_scores(_tokenize(user_question))
        bm25_rank = list(np.argsort(bm25_scores)[::-1][:bm25_top_k])
        fused = _rrf_fuse([faiss_rank, bm25_rank], k=rrf_k)
        top_indices = fused[:top_k]
    else:
        top_indices = faiss_rank[:top_k]

    if manual_index.retrieval_mode == "parent_child":
        parent_ids = []
        for i in top_indices:
            pid = manual_index.chunks[i].parent_id
            if pid not in parent_ids:
                parent_ids.append(pid)
        parent_ids = sorted(parent_ids)
        return [manual_index.parents[pid] for pid in parent_ids]

    top_indices = sorted(top_indices)
    return [manual_index.chunks[i] for i in top_indices]

def _chunk_pages(pages, chunk_size, overlap):
    """Split per-page text into overlapping chunks, tagging each chunk with the
    page number it came from (and a best-effort section header) for debugging."""
    chunks: List[Chunk] = []
    current_header = ""
    for page_number, raw_text in pages:
        current_header = _find_last_header(raw_text, default=current_header)
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            continue

        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(Chunk(text=text[start:end], page=page_number, header=current_header))
            if end == text_len:
                break
            start = end - overlap
    return chunks

def _chunk_pages_parent_child(pages, parent_size, parent_overlap, child_size, child_overlap):
    parents: List[Chunk] = []
    children: List[Chunk] = []
    current_header = ""
    for page_number, raw_text in pages:
        current_header = _find_last_header(raw_text, default=current_header)
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            continue

        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + parent_size, text_len)
            parent_text = text[start:end]
            parent_id = len(parents)
            parents.append(Chunk(text=parent_text, page=page_number, header=current_header))

            # split this parent into smaller children for embedding/search
            c_start = 0
            p_len = len(parent_text)
            while c_start < p_len:
                c_end = min(c_start + child_size, p_len)
                children.append(Chunk(
                    text=parent_text[c_start:c_end],
                    page=page_number,
                    header=current_header,
                    parent_id=parent_id,
                ))
                if c_end == p_len:
                    break
                c_start = c_end - child_overlap

            if end == text_len:
                break
            start = end - parent_overlap
    return parents, children

def build_chunks(document, rag_settings, pdf_reader: str = "pypdf") -> ManualIndex:
    faiss_db_dir = getattr(rag_settings, "faiss_db_dir", None)
    build_faiss_db = getattr(rag_settings, "build_faiss_db", True)

    if not build_faiss_db and faiss_db_dir:
        logging.info("build_faiss_db=false -> loading FAISS DB from '%s'", faiss_db_dir)
        return load_manual_index(faiss_db_dir)

    logging.info("build_faiss_db=true -> reading PDF (pdf_reader='%s') and (re)building the FAISS DB", pdf_reader)
    pages = _extract_pages_from_pdf(document, pdf_reader=pdf_reader)

    if rag_settings.retrieval_mode == "parent_child":
        parents, children = _chunk_pages_parent_child(
            pages,
            rag_settings.parent_chunk_size, rag_settings.parent_chunk_overlap,
            rag_settings.child_chunk_size, rag_settings.child_chunk_overlap,
        )
        index = _build_faiss_index(children, rag_settings.embedding_model) if children else None
        bm25 = _build_bm25_index(children) if rag_settings.search_mode == "hybrid" else None
        manual_index = ManualIndex(chunks=children, parents=parents, index=index, bm25=bm25,
                                    embedding_model=rag_settings.embedding_model,
                                    retrieval_mode="parent_child")
    else:
        chunks = _chunk_pages(
            pages,
            chunk_size=rag_settings.flat_chunk_size,
            overlap=rag_settings.flat_chunk_overlap,
        )
        index = _build_faiss_index(chunks, rag_settings.embedding_model) if chunks else None
        bm25 = _build_bm25_index(chunks) if rag_settings.search_mode == "hybrid" else None
        manual_index = ManualIndex(chunks=chunks, index=index, bm25=bm25,
                                    embedding_model=rag_settings.embedding_model,
                                    retrieval_mode="flat")

    if faiss_db_dir:
        save_manual_index(manual_index, faiss_db_dir)

    return manual_index


def run_rag(client, manual_index: ManualIndex, user_question, top_k,
            search_mode: str = "semantic", bm25_top_k: int = 20, rrf_k: int = 60):
    logging.info("Retrieving chunks relevant to the question (RAG - %s)", search_mode.upper())
    start_time = time.time()
    relevant_chunks = _retrieve_relevant_chunks(
        manual_index, user_question, top_k=top_k,
        search_mode=search_mode, bm25_top_k=bm25_top_k, rrf_k=rrf_k,
    )
    # Debug-only: log where each retrieved chunk came from (page/header), without
    # exposing this to the end user / final answer.
    for c in relevant_chunks:
        logging.info("\t Retrieved chunk [page=%s, header='%s']: %.80s...", c.page, c.header, c.text)

    context = "\n\n---\n\n".join(c.text for c in relevant_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the "
        "provided document context. If the answer is not in the context, say you "
        "don't know."
    )
    user_prompt = (
        f"Context from the document:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        "Answer using only the information in the context above."
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={
            "reasoning_effort": "low",
            "seed": 0,
        }
    )
    answer     = response.choices[0].message.content
    end_time   = time.time()
    logging.info(f"\t Question: {user_question}")
    logging.info(f"\t [{(end_time-start_time):.2f} sec] Answer: {answer}")
    return answer



def run_with_user_question(client, document):
    #user_question = "where is the drive enable switch located ?"
    #user_question = "where is the SGL switch located ?"
    #user_question = "where is the fire enable switch located ?"
    user_question = "How do I get to the Maintenance screen?"
    # user_question = "How do I get to the dog&cat screen?"
    # user_question = "how can I select EOSS FLIR parameters"
    start_time = time.time()
    manual_index = build_chunks(document)
    run_rag(client, manual_index, user_question)
    end_time = time.time()
    logging.info(f"Elapsed time: {(end_time - start_time):.2f} seconds ")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")




    base_url = "http://localhost:8090/v1"
    model = "openai/gpt-oss-20b"

    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url
    )

    document = "../docs/STRYKER-OPERATOR.PDF"


    start_time = time.time()
    run_with_user_question(client, document)
    end_time = time.time()
    logging.info(f"Elapsed time: {(end_time - start_time):.2f} seconds ")

