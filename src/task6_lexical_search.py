"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
bm25_index = None

def load_corpus():
    """Load corpus từ các file markdown."""
    global CORPUS
    if CORPUS:
        return
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        # Tách nội dung thành các đoạn nhỏ (paragraphs) để tìm kiếm tốt hơn
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            if len(p) > 20:  # Bỏ qua các đoạn quá ngắn
                CORPUS.append({
                    "content": p,
                    "metadata": {"source": md_file.name, "type": doc_type, "chunk": i}
                })

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # Tokenize - đơn giản là lowercase và split theo khoảng trắng
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global bm25_index
    if not CORPUS:
        load_corpus()
    if bm25_index is None:
        bm25_index = build_bm25_index(CORPUS)
        
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    # Lấy ra index của top_k documents có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    print("Loading corpus and building index...")
    results = lexical_search("phương thức thanh toán học phí", top_k=5)
    print("\n--- TEST RESULTS ---")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
