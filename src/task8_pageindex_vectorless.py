"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Console Windows mặc định cp1252 → print tiếng Việt sẽ crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_CACHE_DIR = Path(__file__).parent.parent / "data" / "pageindex_pdf"

# doc_id map lưu lại sau khi upload, để pageindex_search() không phải upload lại
# mỗi lần query. File này KHÔNG commit (chứa id riêng của account PageIndex).
DOC_ID_MAP_PATH = PDF_CACHE_DIR / "doc_ids.json"

# Polling khi chờ document xử lý xong (tree generation + OCR) trước khi query được.
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 300


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """PageIndex chỉ nhận PDF — convert markdown thô sang PDF đơn giản bằng fpdf2."""
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    # FPDF core font (Helvetica) chỉ hỗ trợ Latin-1 — corpus có tiếng Việt có dấu,
    # nên thay ký tự ngoài Latin-1 để tránh FPDFException khi ghi PDF.
    safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 6, safe_text)
    pdf.output(str(pdf_path))


def upload_documents() -> dict[str, str]:
    """
    Convert markdown -> PDF rồi upload toàn bộ documents lên PageIndex.

    Returns:
        dict: {md_filename: doc_id} — cũng được lưu vào DOC_ID_MAP_PATH.
    """
    from pageindex.client import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    doc_ids: dict[str, str] = {}
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        pdf_path = PDF_CACHE_DIR / f"{md_file.stem}.pdf"
        _markdown_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        doc_ids[md_file.name] = doc_id
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    DOC_ID_MAP_PATH.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc_ids


def _load_doc_ids() -> dict[str, str]:
    if not DOC_ID_MAP_PATH.exists():
        return {}
    return json.loads(DOC_ID_MAP_PATH.read_text(encoding="utf-8"))


def _wait_until_doc_ready(client, doc_id: str) -> None:
    """Đợi PageIndex xử lý xong tree generation + OCR cho document trước khi query."""
    start = time.time()
    while time.time() - start < POLL_TIMEOUT_SECONDS:
        if client.is_retrieval_ready(doc_id):
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Document {doc_id} chưa sẵn sàng sau {POLL_TIMEOUT_SECONDS}s")


def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    """
    Poll get_retrieval() tới khi có kết quả.

    Lưu ý: SDK không expose enum trạng thái chính thức (get_retrieval() chỉ trả
    nguyên response.json() của API). Coi là "xong" khi response đã có key
    'retrieved_nodes' — đúng schema ghi nhận trong docstring module này. Nếu
    schema thật khác (API đổi), block dưới sẽ tự in raw response ra để chỉnh lại
    logic parse thay vì fail âm thầm.
    """
    start = time.time()
    last_response: dict = {}
    while time.time() - start < POLL_TIMEOUT_SECONDS:
        last_response = client.get_retrieval(retrieval_id)
        if last_response.get("deprecation"):
            print(f"  ⚠ PageIndex API cảnh báo deprecation: {last_response['deprecation']}")
        if "retrieved_nodes" in last_response:
            return last_response
        time.sleep(POLL_INTERVAL_SECONDS)

    print("  ⚠ Không thấy 'retrieved_nodes' trong response — raw response để debug:")
    print(json.dumps(last_response, ensure_ascii=False, indent=2)[:2000])
    raise TimeoutError(f"Retrieval {retrieval_id} không có kết quả sau {POLL_TIMEOUT_SECONDS}s")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    doc_ids = _load_doc_ids()
    if not doc_ids:
        raise RuntimeError(
            f"Chưa có document nào được upload — chạy upload_documents() trước "
            f"(không tìm thấy {DOC_ID_MAP_PATH})"
        )

    # SDK chỉ hỗ trợ query theo từng doc_id (không có endpoint query toàn corpus),
    # nên phải lặp qua từng document đã upload rồi gộp kết quả lại.
    results = []
    for filename, doc_id in doc_ids.items():
        _wait_until_doc_ready(client, doc_id)

        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        retrieval = _wait_for_retrieval(client, retrieval_id)

        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "metadata": {
                            "section": item.get("section_title"),
                            "source_file": filename,
                        },
                        "source": "pageindex",
                    })

    # PageIndex không trả score trực tiếp — tự gán theo rank (kết quả đầu ưu tiên hơn).
    for i, item in enumerate(results):
        item["score"] = round(max(0.1, 1.0 - i * 0.1), 2)

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
