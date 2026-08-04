"""
RAG Chatbot — University Services (Yêu cầu 1 của bài tập nhóm).

Streamlit UI nối Retrieval (Task 9) + Generation có citation (Task 10), bổ sung
conversation memory để trả lời được câu hỏi follow-up.

Chạy:
    streamlit run app.py

Luồng xử lý 1 lượt hỏi:
    câu hỏi mới + lịch sử hội thoại
      └→ condense_question()          # LLM viết lại thành câu hỏi ĐỘC LẬP
           └→ generate_with_citation() # Task 9 retrieval → Task 10 generation
                └→ answer + sources hiển thị trên UI

Vì sao phải viết lại câu hỏi: retrieval là stateless. Câu follow-up kiểu "vậy còn
sinh viên quốc tế thì sao?" đem đi embed trực tiếp sẽ lấy về chunk rác — không có
từ khoá "học bổng"/"học phí" nào trong đó. Nén lịch sử vào 1 câu hỏi độc lập trước
khi retrieve là cách xử lý chuẩn (history-aware retriever).
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Số lượt hỏi–đáp gần nhất đưa vào bộ nhớ hội thoại. Để nhỏ (3) vì lịch sử càng dài
# thì câu hỏi viết lại càng dễ bị "trôi" sang chủ đề cũ.
MEMORY_WINDOW_DEFAULT = 3

# Cắt bớt mỗi lượt trong lịch sử để prompt condense không phình theo độ dài câu trả lời.
HISTORY_CHAR_LIMIT = 500

CONDENSE_SYSTEM_PROMPT = """Bạn là bộ viết lại câu hỏi cho hệ thống tìm kiếm tài liệu.

Nhiệm vụ: dựa vào lịch sử hội thoại, viết lại câu hỏi mới nhất thành MỘT câu hỏi độc lập,
đầy đủ ngữ cảnh, người đọc hiểu được mà không cần xem lịch sử.

Quy tắc:
1. Thay mọi tham chiếu mơ hồ ("cái đó", "vậy còn", "nó", "trường hợp này", "mức đó")
   bằng đối tượng cụ thể đã được nhắc trong lịch sử
2. Giữ nguyên ngôn ngữ của câu hỏi gốc (tiếng Việt)
3. Nếu câu hỏi đã độc lập rồi → trả về Y NGUYÊN, không thêm bớt
4. CHỈ xuất ra câu hỏi, không giải thích, không thêm dấu ngoặc kép"""

SUGGESTIONS = [
    "Học phí chương trình Cử nhân ở VinUni là bao nhiêu?",
    "Học bổng năm học 2026-2027 có những mức nào?",
    "Phí phạt mượn tài liệu thư viện quá hạn tính thế nào?",
    "Chương trình hỗ trợ tài chính cho sinh viên đang học gồm những gì?",
    "Điều kiện tuyển sinh sau đại học trình độ thạc sĩ là gì?",
]


# =============================================================================
# HELPERS
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_llm_client():
    """Client cho bước condense — ưu tiên OpenRouter, fallback OpenAI (giống Task 10)."""
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openrouter_key:
        from src.task10_generation import LLM_MODEL

        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"), LLM_MODEL
    if openai_key:
        return OpenAI(api_key=openai_key), os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return None, None


@st.cache_data(show_spinner=False, ttl=120)
def get_corpus_status() -> dict:
    """Đếm số chunk trong ChromaDB để cảnh báo sớm nếu chưa chạy Task 4."""
    try:
        from src.task4_chunking_indexing import COLLECTION_NAME, get_collection

        return {"ok": True, "count": get_collection().count(), "collection": COLLECTION_NAME}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
def condense_question(question: str, history: list[dict], window: int) -> tuple[str, bool]:
    """
    Viết lại câu hỏi follow-up thành câu hỏi độc lập dựa trên lịch sử hội thoại.

    Returns:
        (câu hỏi dùng để retrieve, đã bị viết lại hay chưa)
    """
    if not history:
        return question, False

    client, model = get_llm_client()
    if client is None:
        return question, False

    turns = []
    for msg in history[-window * 2:]:
        speaker = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        turns.append(f"{speaker}: {msg['content'][:HISTORY_CHAR_LIMIT]}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Lịch sử hội thoại:\n"
                    + "\n".join(turns)
                    + f"\n\nCâu hỏi mới nhất: {question}\n\nCâu hỏi độc lập:",
                },
            ],
            temperature=0,
            max_tokens=150,
        )
        rewritten = (response.choices[0].message.content or "").strip().strip('"')
    except Exception:
        # Condense lỗi (rate limit / mạng) thì vẫn phải trả lời được — dùng câu gốc.
        return question, False

    if not rewritten:
        return question, False
    return rewritten, rewritten.lower() != question.strip().lower()


def render_sources(sources: list[dict]) -> None:
    """Hiển thị các chunk đã dùng để trả lời (yêu cầu 'show source documents')."""
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata") or {}
            # metadata khác nhau giữa hybrid (source/type/year) và PageIndex (section/source_file)
            name = meta.get("source") or meta.get("source_file") or meta.get("title") or "Unknown"
            doc_type = meta.get("type") or meta.get("section") or "unknown"
            st.markdown(
                f"**[{i}] {name}** `{doc_type}` · score `{src.get('score', 0.0):.4f}`"
            )
            st.text((src.get("content") or "")[:300] + "...")
            if i < len(sources):
                st.divider()


def render_meta(meta: dict) -> None:
    """Dòng thông tin nhỏ dưới câu trả lời: nguồn retrieval, thời gian, câu hỏi đã viết lại."""
    if not meta:
        return
    if meta.get("rewritten_from"):
        st.caption(f"🔁 Hiểu câu hỏi thành: *{meta['standalone_query']}*")
    bits = []
    if meta.get("retrieval_source"):
        bits.append(f"retrieval: `{meta['retrieval_source']}`")
    if meta.get("n_sources") is not None:
        bits.append(f"{meta['n_sources']} chunks")
    if meta.get("latency") is not None:
        bits.append(f"{meta['latency']:.1f}s")
    if bits:
        st.caption(" · ".join(bits))


def answer_query(query: str, top_k: int, use_memory: bool, window: int) -> dict:
    """Chạy 1 lượt hỏi đáp đầy đủ: condense (nếu bật) → retrieve → generate."""
    from src.task10_generation import generate_with_citation

    history = st.session_state.messages if use_memory else []
    standalone, rewritten = condense_question(query, history, window)

    started = time.time()
    response = generate_with_citation(standalone, top_k=top_k)
    sources = response.get("sources", [])

    return {
        "answer": response.get("answer", "Chưa thể trả lời."),
        "sources": sources,
        "meta": {
            "standalone_query": standalone,
            "rewritten_from": query if rewritten else None,
            "retrieval_source": response.get("retrieval_source", "none"),
            "n_sources": len(sources),
            "latency": time.time() - started,
        },
    }
# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# =============================================================================
# SIDEBAR — SETTINGS & INFO
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption(
        "Trợ lý hỏi đáp về dịch vụ và chính sách đại học "
        "(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần)"
    )

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    for s in SUGGESTIONS:
        if st.button(s, use_container_width=True, key=f"sug_{s[:24]}"):
            st.session_state["pending_query"] = s
    st.caption(
        "Demo follow-up: hỏi câu 1, rồi hỏi tiếp *“vậy còn ngành Y khoa thì sao?”* "
        "— hệ thống tự hiểu là đang nói về học phí."
    )

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    use_memory = st.toggle(
        "Ghi nhớ hội thoại (follow-up)",
        value=True,
        help="Bật để câu hỏi kiểu 'vậy còn sinh viên quốc tế thì sao?' được hiểu "
        "theo ngữ cảnh các lượt trước.",
    )
    memory_window = st.slider(
        "Số lượt nhớ lại", 1, 5, MEMORY_WINDOW_DEFAULT, disabled=not use_memory
    )

    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.subheader("📊 Trạng thái hệ thống")
    status = get_corpus_status()
    if status["ok"] and status["count"] > 0:
        st.success(f"ChromaDB: {status['count']} chunks")
    elif status["ok"]:
        st.error("ChromaDB rỗng — chạy `python -m src.task4_chunking_indexing` trước.")
    else:
        st.error(f"Không đọc được vector store: {status['error']}")

    has_llm_key = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    st.write("LLM key:", "✅" if has_llm_key else "❌ thiếu trong .env")
    st.write("PageIndex fallback:", "✅" if os.getenv("PAGEINDEX_API_KEY") else "➖ tắt")

    st.divider()
    st.caption("**Kiến trúc:**")
    st.caption(
        "Condense follow-up → Hybrid Retrieval (Semantic + BM25) → RRF Rerank "
        "→ PageIndex Fallback → LLM Generation có Citation"
    )
# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption(
    "Hỏi đáp chính sách & dịch vụ đại học — trả lời kèm citation, có nhớ ngữ cảnh hội thoại"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(msg.get("meta"))
            render_sources(msg.get("sources", []))

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                result = answer_query(query, top_k, use_memory, memory_window)
            except NotImplementedError:
                result = {
                    "answer": "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành "
                    "`src/task10_generation.py` để kết nối pipeline vào UI!",
                    "sources": [],
                    "meta": {},
                }
            except Exception as exc:
                result = {
                    "answer": f"❌ **Lỗi khi chạy RAG Pipeline:** `{type(exc).__name__}` {exc}",
                    "sources": [],
                    "meta": {},
                }

        st.markdown(result["answer"])
        render_meta(result["meta"])
        render_sources(result["sources"])

    # Lịch sử chỉ append SAU khi đã trả lời, để condense_question() ở lượt này không
    # nhìn thấy chính câu hỏi đang xử lý.
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "meta": result["meta"],
        }
    )

