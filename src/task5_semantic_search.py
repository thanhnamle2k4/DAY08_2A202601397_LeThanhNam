"""
Task 5 — Semantic Search Module (Dense Retrieval + HyDE).

Tìm kiếm ngữ nghĩa trên ChromaDB đã index ở Task 4, xếp hạng theo Cosine Similarity.

Hai hàm chính:
    - semantic_search()       → dense retrieval thuần (bắt buộc)
    - semantic_search_hyde()  → HyDE: cho LLM sinh đoạn trả lời giả định rồi mới
                                embed, giúp query ngắn/khác ngôn ngữ khớp tốt hơn
                                với văn phong văn bản quy chế (điểm bonus)

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Dùng LẠI embed_query() và get_collection() của Task 4 để chắc chắn cùng
      embedding model và cùng collection — lệch model là lệch số chiều vector,
      ChromaDB sẽ báo lỗi ngay.
"""

import os
import sys

from dotenv import load_dotenv

try:
    from .task4_chunking_indexing import embed_query, get_collection
except ImportError:
    from task4_chunking_indexing import embed_query, get_collection

load_dotenv()

# Console Windows mặc định cp1252 → print tiếng Việt sẽ crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# CONFIGURATION — HyDE
# =============================================================================

# LLM sinh đoạn giả định. Dùng model nhỏ/rẻ vì đoạn giả định chỉ cần đúng văn phong
# và đúng từ khoá chuyên ngành, không cần chính xác về nội dung.
HYDE_LLM_MODEL = "gpt-4o-mini"
HYDE_MAX_TOKENS = 220
HYDE_TEMPERATURE = 0.3

# Bắt LLM viết bằng TIẾNG VIỆT: corpus là văn bản quy chế tiếng Việt, nên đoạn giả
# định tiếng Việt sẽ nằm gần các chunk thật trong không gian vector hơn nhiều.
HYDE_SYSTEM_PROMPT = """Bạn viết một đoạn văn bản GIẢ ĐỊNH mô phỏng nội dung một
điều/khoản trong quy chế, quy định của trường đại học Việt Nam.

Quy tắc:
- Viết bằng tiếng Việt, 2-4 câu, văn phong hành chính như văn bản quy định thật.
- Dùng đúng thuật ngữ chuyên ngành liên quan tới câu hỏi (học phí, học bổng, tín chỉ,
  CGPA, học vụ, xét tuyển...).
- KHÔNG mở đầu bằng "Tôi không biết" hay bất kỳ lời rào đón nào.
- Nội dung có thể không chính xác — đoạn này chỉ dùng để tìm kiếm, không dùng để trả lời."""


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (0..1, càng cao càng khớp)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    collection = get_collection()
    total = collection.count()
    if total == 0:
        # Chưa index → trả list rỗng thay vì raise, để pipeline Task 9 còn fallback được.
        return []

    results = collection.query(
        query_embeddings=[embed_query(query)],
        # Xin nhiều hơn số chunk đang có sẽ khiến Chroma cảnh báo → clamp lại.
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Collection tạo với hnsw:space="cosine" → distance = 1 - cosine_similarity.
        # Clamp về [0, 1] vì sai số dấu phẩy động có thể cho ra 1.0000001 hoặc -1e-7.
        score = min(1.0, max(0.0, 1.0 - float(dist)))
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": dict(meta),
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


def generate_hypothetical_document(query: str) -> str | None:
    """
    HyDE bước 1: cho LLM sinh một đoạn văn bản giả định trả lời câu hỏi.

    Returns:
        Đoạn giả định, hoặc None nếu không có API key.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=HYDE_LLM_MODEL,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=HYDE_TEMPERATURE,
        max_tokens=HYDE_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip() or None


def semantic_search_hyde(query: str, top_k: int = 10) -> list[dict]:
    """
    HyDE (Hypothetical Document Embeddings) — dense retrieval nâng cao.

    Vì sao cần? Query của người dùng thường ngắn và dùng từ ngữ đời thường
    ("đóng học phí muộn bị sao?"), trong khi chunk trong index là văn bản hành chính
    ("Sinh viên không hoàn thành nghĩa vụ tài chính trong thời hạn quy định..."). Embed
    trực tiếp query ngắn cho score thấp. HyDE cho LLM viết trước một đoạn giả định
    ĐÚNG VĂN PHONG tài liệu, rồi embed đoạn đó → vector nằm gần chunk thật hơn.

    Ghép cả query gốc + đoạn giả định khi embed, để nếu LLM sinh lệch chủ đề thì query
    gốc vẫn neo lại đúng hướng tìm kiếm.

    Nếu LLM lỗi (hết quota, mất mạng) → tự động fallback về semantic_search() thuần.
    """
    try:
        hypothetical = generate_hypothetical_document(query)
    except Exception as exc:
        print(f"  ⚠ HyDE lỗi ({type(exc).__name__}) → fallback dense search thuần")
        hypothetical = None

    if not hypothetical:
        return semantic_search(query, top_k)

    return semantic_search(f"{query}\n{hypothetical}", top_k)


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
