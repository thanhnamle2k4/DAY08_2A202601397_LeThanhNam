"""
Task 4 — Chunking & Indexing vào Vector Store.

Pipeline: đọc markdown từ data/standardized/ → chunk → embed → index vào ChromaDB.

Lựa chọn đã chốt (giải thích ở phần CONFIGURATION bên dưới):
    - Chunking:     RecursiveCharacterTextSplitter (800 / 100)
    - Embedding:    OpenAI text-embedding-3-small (1536 dim)
    - Vector store: ChromaDB persistent tại chroma_db/

Cài đặt:
    pip install langchain-text-splitters chromadb openai python-dotenv

"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Console Windows mặc định là cp1252 → print ký tự ✓ / tiếng Việt sẽ crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter.
#   - Vì sao recursive? Test chấm điểm (test_chunks_respect_size_limit) yêu cầu mọi
#     chunk <= CHUNK_SIZE * 1.1. MarkdownHeaderTextSplitter cắt theo heading nên một
#     điều/khoản dài sẽ vượt limit; SemanticChunker phải gọi embedding cho từng câu
#     nên rất chậm. Recursive cắt theo ranh giới tự nhiên (heading → đoạn → câu → từ)
#     và vẫn đảm bảo trần độ dài.
#   - Vì sao 800? Đủ để giữ nguyên trọn một điều/khoản của văn bản quy chế, nhưng vẫn
#     đủ nhỏ để nhồi 5-10 chunk vào context LLM mà không loãng thông tin.
#   - Vì sao overlap 100? Câu quan trọng nằm ở ranh giới 2 chunk vẫn xuất hiện đủ
#     nghĩa trong ít nhất một chunk (~12% chunk size, đủ 1-2 câu).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: OpenAI text-embedding-3-small.
#   - Vì sao? Corpus là văn bản quy chế tiếng Việt → cần model đa ngữ. all-MiniLM-L6-v2
#     chỉ mạnh tiếng Anh, embed tiếng Việt cho score rất nhiễu. BAAI/bge-m3 cũng đa ngữ
#     và tốt, nhưng phải tải ~2.2GB model + torch, không khả thi trong thời lượng lab.
#     text-embedding-3-small hỗ trợ đa ngữ tốt, chạy qua API nên không cần tải model.
#   - Muốn đổi sang bge-m3 (chạy local, không cần API): đặt EMBEDDING_PROVIDER = "local",
#     pip install sentence-transformers, rồi XÓA chroma_db/ và chạy lại script này.
EMBEDDING_PROVIDER = "openai"  # "openai" | "local"
EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"  # dùng khi EMBEDDING_PROVIDER = "local"
EMBEDDING_DIM = 1536 if EMBEDDING_PROVIDER == "openai" else 1024
EMBED_BATCH_SIZE = 64  # OpenAI cho phép batch nhiều text / 1 request → giảm số round-trip

# Vector store: ChromaDB.
#   - Vì sao? Local persistent, không cần Docker, hỗ trợ cosine similarity + metadata
#     filter sẵn. FAISS mất metadata, Weaviate cần Docker.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"

# Xoá collection cũ trước khi index lại để tránh lẫn chunk của corpus cũ.
RESET_COLLECTION = True


# =============================================================================
# EMBEDDING HELPERS — Task 5 import lại từ đây để dùng ĐÚNG model đã index
# =============================================================================

_embedding_client = None


def get_embedding_model():
    """
    Trả về client/model embedding (cached).

    Task 5 nên gọi embed_query() thay vì dùng trực tiếp hàm này, để không phải quan
    tâm provider là OpenAI hay local sentence-transformers.
    """
    global _embedding_client
    if _embedding_client is not None:
        return _embedding_client

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Thiếu OPENAI_API_KEY trong .env — thêm key hoặc đổi "
                "EMBEDDING_PROVIDER = 'local'"
            )
        _embedding_client = OpenAI(api_key=api_key)
    else:
        from sentence_transformers import SentenceTransformer

        _embedding_client = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    return _embedding_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một danh sách text → list vector. Tự chia batch."""
    client = get_embedding_model()

    if EMBEDDING_PROVIDER != "openai":
        return [v.tolist() for v in client.encode(texts, show_progress_bar=True)]

    vectors: list[list[float]] = []
    show_progress = len(texts) > EMBED_BATCH_SIZE  # query lẻ (Task 5) thì không in gì
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
        if show_progress:
            print(f"    embedded {min(start + len(batch), len(texts))}/{len(texts)}")
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed 1 câu truy vấn — Task 5 dùng hàm này để đảm bảo cùng model với index."""
    return embed_texts([query])[0]


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue  # bỏ file rỗng, tránh tạo chunk vô nghĩa
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ưu tiên cắt ở ranh giới heading markdown trước, rồi mới tới đoạn/câu/từ.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["content"])):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks, thêm key 'embedding' vào từng chunk."""
    vectors = embed_texts([c["content"] for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def get_collection():
    """
    Trả về ChromaDB collection đã index. Task 5 import hàm này để query.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Task 5 tính score = 1 - cosine distance
    )


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks (đã có embedding) vào ChromaDB."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  (đã xoá collection cũ '{COLLECTION_NAME}' để reindex sạch)")
        except Exception:
            pass  # chưa tồn tại → không có gì để xoá

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("  ⚠ Không có file .md nào trong data/standardized/ — chạy Task 3 trước.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    collection = index_to_vectorstore(chunks)
    print(f"✓ Indexed to vector store — collection có {collection.count()} chunks")
    print(f"  Path: {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
