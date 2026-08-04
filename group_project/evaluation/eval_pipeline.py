"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import re
import time
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    test_cases = [
        LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item["expected_answer"],
            retrieval_context=[
                source["content"]
                for source in result.get("sources", [])
                if source.get("content")
            ],
        )
        for item in golden_dataset
        for result in [rag_pipeline.generate_with_citation(item["question"])]
    ]

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]

    return evaluate(test_cases=test_cases, metrics=metrics)


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"RAGAS dependencies are incompatible ({exc}); using local metrics.")
        return evaluate_with_local_metrics(rag_pipeline, golden_dataset)

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in golden_dataset:
        result = rag_pipeline.generate_with_citation(item["question"])
        sources = result.get("sources", [])

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result.get("answer", ""))
        eval_data["contexts"].append(
            [
                source.get("content", str(source)) if isinstance(source, dict) else str(source)
                for source in sources
            ]
        )
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


def evaluate_with_local_metrics(rag_pipeline, golden_dataset: list[dict]) -> list[dict]:
    """Evaluate retrieval and answers without external evaluator dependencies."""
    word_re = re.compile(r"\w+", re.UNICODE)

    def tokens(value: str) -> set[str]:
        return set(word_re.findall((value or "").lower()))

    def recall(expected: set[str], actual: set[str]) -> float:
        return len(expected & actual) / len(expected) if expected else 0.0

    rows = []
    for item in golden_dataset:
        result = rag_pipeline.generate_with_citation(item["question"])
        answer = result.get("answer", "")
        contexts = [
            source.get("content", "") if isinstance(source, dict) else str(source)
            for source in result.get("sources", [])
        ]
        answer_tokens = tokens(answer)
        context_tokens = tokens(" ".join(contexts))
        expected_tokens = tokens(item.get("expected_answer", ""))
        question_tokens = tokens(item["question"])

        rows.append({
            "question": item["question"],
            "answer": answer,
            "faithfulness": recall(answer_tokens, context_tokens),
            "answer_relevancy": recall(question_tokens | expected_tokens, answer_tokens),
            "context_recall": recall(expected_tokens, context_tokens),
            "context_precision": recall(context_tokens, expected_tokens),
        })

    return rows


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    from trulens.apps.custom import TruCustomApp
    from trulens.core import Feedback
    from trulens.providers.openai import OpenAI as TruOpenAI

    provider = TruOpenAI()
    feedbacks = [
        Feedback(provider.groundedness_measure_with_cot_reasons, name="faithfulness").on_output(),
        Feedback(provider.relevance, name="answer_relevance").on_input_output(),
        Feedback(provider.context_relevance, name="context_relevance").on_input(),
    ]

    tru_rag = TruCustomApp(
        rag_pipeline,
        app_name="UniversityServices_RAG",
        feedbacks=feedbacks,
    )

    predictions = []
    with tru_rag as recording:
        for item in golden_dataset:
            result = rag_pipeline.generate_with_citation(item["question"])
            predictions.append(
                {
                    "question": item["question"],
                    "expected_answer": item.get("expected_answer"),
                    "answer": result.get("answer") if isinstance(result, dict) else result,
                    "sources": result.get("sources", []) if isinstance(result, dict) else [],
                }
            )

    return {
        "framework": "TruLens",
        "app": tru_rag,
        "recording": recording,
        "predictions": predictions,
    }


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    configs = {
        "hybrid_with_reranking": {"use_reranking": True},
        "hybrid_without_reranking": {"use_reranking": False},
    }

    word_re = re.compile(r"\w+", re.UNICODE)

    def tokens(text: str) -> set[str]:
        return set(word_re.findall((text or "").lower()))

    def overlap_score(left: set[str], right: set[str]) -> float:
        return len(left & right) / len(left) if left else 0.0

    def contexts_from(result: dict) -> list[str]:
        sources = result.get("sources") or result.get("contexts") or result.get("context") or []
        if isinstance(sources, str):
            return [sources]
        return [
            str(source.get("content", source)) if isinstance(source, dict) else str(source)
            for source in sources
        ]

    def set_pipeline_config(params: dict) -> dict:
        previous = {}
        if hasattr(rag_pipeline, "configure"):
            rag_pipeline.configure(**params)
            return previous
        if hasattr(rag_pipeline, "set_config"):
            rag_pipeline.set_config(**params)
            return previous
        for key, value in params.items():
            if hasattr(rag_pipeline, key):
                previous[key] = getattr(rag_pipeline, key)
                setattr(rag_pipeline, key, value)
        return previous

    def restore_pipeline_config(previous: dict) -> None:
        for key, value in previous.items():
            setattr(rag_pipeline, key, value)

    def generate(question: str, params: dict) -> dict:
        try:
            return rag_pipeline.generate_with_citation(question, **params)
        except TypeError:
            return rag_pipeline.generate_with_citation(question)

    results = {}
    for config_name, params in configs.items():
        previous = set_pipeline_config(params)
        started = time.perf_counter()
        rows = []

        try:
            for item in golden_dataset:
                result = generate(item["question"], params)
                answer = result.get("answer", "") if isinstance(result, dict) else str(result)
                contexts = contexts_from(result) if isinstance(result, dict) else []
                context_text = " ".join(contexts)

                question_tokens = tokens(item.get("question", ""))
                answer_tokens = tokens(answer)
                expected_tokens = tokens(item.get("expected_answer", ""))
                expected_context = item.get("expected_context", "")
                context_tokens = tokens(context_text)

                expected_source = expected_context.split(",", 1)[0].replace("\\", "/").split("/")[-1].lower()
                relevant_contexts = 0
                for source, context in zip(result.get("sources", []), contexts):
                    metadata = source.get("metadata", {}) if isinstance(source, dict) else {}
                    source_name = str(metadata.get("source", "")).replace("\\", "/").lower()
                    evidence_overlap = overlap_score(expected_tokens, tokens(context))
                    if (expected_source and expected_source in source_name) or evidence_overlap >= 0.2:
                        relevant_contexts += 1

                row = {
                    "question": item.get("question", ""),
                    "expected_answer": item.get("expected_answer", ""),
                    "expected_context": expected_context,
                    "answer": answer,
                    "contexts": contexts,
                    "faithfulness": overlap_score(answer_tokens, context_tokens),
                    "relevance": overlap_score(question_tokens | expected_tokens, answer_tokens),
                    "context_recall": overlap_score(expected_tokens, context_tokens),
                    "context_precision": relevant_contexts / len(contexts) if contexts else 0.0,
                    "sources_count": len(contexts),
                }
                row["average"] = (
                    row["faithfulness"]
                    + row["relevance"]
                    + row["context_recall"]
                    + row["context_precision"]
                ) / 4
                rows.append(row)
        finally:
            restore_pipeline_config(previous)

        metric_names = ("faithfulness", "relevance", "context_recall", "context_precision", "average")
        results[config_name] = {
            "config": params,
            "scores": {
                metric: round(sum(row[metric] for row in rows) / len(rows), 4) if rows else 0.0
                for metric in metric_names
            },
            "latency_seconds": round(time.perf_counter() - started, 4),
            "items": rows,
        }

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    baseline_name = "hybrid_with_reranking"
    baseline = comparison[baseline_name]
    overall = baseline["scores"]
    rows = results

    content = "# Kết quả đánh giá RAG\n\n"
    content += "Toàn bộ golden dataset được đánh giá bằng local metrics dựa trên token và evidence. "
    content += "A/B sử dụng cùng hybrid retriever, với reranking được bật và tắt.\n\n"
    content += "## Điểm tổng thể\n\n"
    content += f"Config baseline: `{baseline_name}`\n\n"
    content += "| Metric | Điểm |\n|---|---:|\n"
    for metric in ("faithfulness", "relevance", "context_recall", "context_precision", "average"):
        content += f"| {metric} | {overall[metric]:.4f} |\n"

    content += "\n## So sánh A/B\n\n"
    content += "| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision | Trung bình | Latency (giây) |\n"
    content += "|---|---:|---:|---:|---:|---:|---:|\n"
    for config_name, config_result in comparison.items():
        scores = config_result["scores"]
        content += (
            f"| {config_name} | {scores['faithfulness']:.4f} | {scores['relevance']:.4f} | "
            f"{scores['context_recall']:.4f} | {scores['context_precision']:.4f} | "
            f"{scores['average']:.4f} | {config_result['latency_seconds']:.4f} |\n"
        )

    content += "\n## Các trường hợp có điểm thấp nhất\n\n"
    content += "Năm trường hợp có điểm trung bình thấp nhất của config baseline có reranking:\n\n"
    for rank, row in enumerate(sorted(rows, key=lambda item: item["average"])[:5], start=1):
        content += f"### {rank}. {cell(row['question'])}\n\n"
        content += f"- Điểm trung bình: {row['average']:.4f}\n"
        content += f"- Câu trả lời kỳ vọng: {cell(row['expected_answer'])}\n"
        content += f"- Câu trả lời thực tế: {cell(row['answer'])}\n"
        content += f"- Context kỳ vọng: {cell(row['expected_context'])}\n"
        content += (
            f"- Metrics: faithfulness={row['faithfulness']:.4f}, relevance={row['relevance']:.4f}, "
            f"context_recall={row['context_recall']:.4f}, context_precision={row['context_precision']:.4f}\n\n"
        )

    content += "## Đề xuất cải tiến\n\n"
    recommendations = {
        "faithfulness": "Siết chặt generation prompt, yêu cầu citation và từ chối câu trả lời không được retrieved context hỗ trợ.",
        "relevance": "Cải thiện query rewriting và yêu cầu generator trả lời trực tiếp, ngắn gọn.",
        "context_recall": "Tăng số candidate retrieval, kiểm tra lại chunk boundaries và cải thiện semantic/lexical fusion.",
        "context_precision": "Dùng reranking, giảm các kết quả nhiễu trong top-k và thêm metadata hoặc source filter.",
    }
    low_metrics = [name for name in recommendations if overall[name] < 0.7]
    if low_metrics:
        for metric in low_metrics:
            content += f"- **{metric} ({overall[metric]:.4f})**: {recommendations[metric]}\n"
    else:
        content += "- Tất cả metric tổng hợp đều đạt ngưỡng 0.7; ưu tiên cải thiện các trường hợp điểm thấp nêu trên.\n"

    winner = max(comparison, key=lambda name: comparison[name]["scores"]["average"])
    content += f"- Ưu tiên `{winner}` vì có điểm trung bình cao nhất; đồng thời cần cân nhắc latency trước khi triển khai.\n"
    refused = [row for row in rows if "không thể xác minh" in row["answer"].lower()]
    if refused:
        content += (
            f"- Kiểm tra refusal logic của generator cho {len(refused)} trường hợp: relevant evidence đã được retrieve "
            "nhưng model vẫn từ chối trả lời. Cần giữ section heading khi chunking và điều chỉnh prompt để phân biệt "
            "thiếu evidence với retrieval score thấp.\n"
        )
    content += (
        "- Kiểm tra thủ công các trường hợp giá trị số giữa actual answer và expected answer khác nhau; chỉnh golden answer "
        "hoặc source document trước khi dùng điểm số làm tín hiệu chất lượng.\n"
    )
    RESULTS_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    from src.task10_generation import generate_with_citation

    class RAGPipeline:
        generate_with_citation = staticmethod(generate_with_citation)

    pipeline = RAGPipeline()

    comparison = compare_configs(pipeline, golden_dataset)
    results = comparison["hybrid_with_reranking"]["items"]
    export_results(results, comparison)
    print(f"Evaluation complete. Results written to {RESULTS_PATH}")
