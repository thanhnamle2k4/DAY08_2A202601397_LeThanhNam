# Kết quả đánh giá RAG

Toàn bộ golden dataset được đánh giá bằng local metrics dựa trên token và evidence. A/B sử dụng cùng hybrid retriever, với reranking được bật và tắt.

## Điểm tổng thể

Config baseline: `hybrid_with_reranking`

| Metric | Điểm |
|---|---:|
| faithfulness | 0.8718 |
| relevance | 0.7363 |
| context_recall | 0.9534 |
| context_precision | 0.9900 |
| average | 0.8879 |

## So sánh A/B

| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision | Trung bình | Latency (giây) |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_with_reranking | 0.8718 | 0.7363 | 0.9534 | 0.9900 | 0.8879 | 44.6850 |
| hybrid_without_reranking | 0.8690 | 0.7298 | 0.9534 | 0.9900 | 0.8856 | 49.5788 |

Reranking cải thiện điểm trung bình thêm 0.0023 và nhanh hơn 4.8938 giây trong lần chạy này. Mức tăng chất lượng khá nhỏ, vì vậy cần đo latency qua nhiều lần chạy trước khi quyết định triển khai.

## Các trường hợp có điểm thấp nhất

Năm trường hợp có điểm trung bình thấp nhất của config baseline có reranking:

### 1. Sinh viên cần hoàn thành nghĩa vụ tài chính vào thời điểm nào để được xét tốt nghiệp?

- Điểm trung bình: 0.5990
- Câu trả lời kỳ vọng: Sinh viên cần hoàn thành mọi nghĩa vụ tài chính trước khi được xét tốt nghiệp.
- Câu trả lời thực tế: Tôi không thể xác minh thông tin này từ nguồn hiện có.
- Context kỳ vọng: data/standardized/legal/quy-dinh-tai-chinh.md, mục nghĩa vụ tài chính trước xét tốt nghiệp
- Metrics: faithfulness=0.5833, relevance=0.0000, context_recall=0.8125, context_precision=1.0000

### 2. Khi nào sinh viên phải xin đăng ký học vượt ngưỡng?

- Điểm trung bình: 0.6010
- Câu trả lời kỳ vọng: Sinh viên phải xin đăng ký học vượt ngưỡng nếu năm thứ nhất muốn học nhiều hơn 18 tín chỉ, nếu không có thành tích tốt ở học kỳ trước nhưng muốn học nhiều hơn 18 tín chỉ, hoặc nếu muốn học nhiều hơn 22 tín chỉ trong kỳ học thông thường.
- Câu trả lời thực tế: Tôi không thể xác minh thông tin này từ nguồn hiện có.
- Context kỳ vọng: data/standardized/legal/quy-che-dao-tao-he-chinh-quy.md, Điều 10 về học vượt ngưỡng
- Metrics: faithfulness=0.5000, relevance=0.0857, context_recall=0.8182, context_precision=1.0000

### 3. Sinh viên chính quy cần đăng ký tối thiểu bao nhiêu tín chỉ để duy trì trạng thái toàn thời gian?

- Điểm trung bình: 0.8185
- Câu trả lời kỳ vọng: Sinh viên phải đăng ký tối thiểu theo quy định của chương trình; tài liệu nêu mức tối thiểu trong mỗi kỳ chính và thông thường là ít nhất 14 tín chỉ cho mỗi học kỳ trong một số trường hợp.
- Câu trả lời thực tế: Sinh viên chính quy cần đăng ký tối thiểu 12 tín chỉ để duy trì trạng thái toàn thời gian. Điều này tương đương với việc đăng ký tối thiểu 80% khối lượng học tập trong một học kỳ thông thường [Quy chế đào tạo hệ chính quy, Document 5].
- Context kỳ vọng: data/standardized/legal/quy-che-dao-tao-he-chinh-quy.md, Điều 10 về khối lượng học tập
- Metrics: faithfulness=0.8810, relevance=0.5319, context_recall=0.8611, context_precision=1.0000

### 4. Điểm trung bình tích lũy tối thiểu để xét tốt nghiệp là bao nhiêu?

- Điểm trung bình: 0.8271
- Câu trả lời kỳ vọng: Điểm trung bình tích lũy của toàn khóa học phải đạt tối thiểu 2,00 trên thang 4,00.
- Câu trả lời thực tế: Điểm trung bình tích lũy tối thiểu để xét tốt nghiệp là 2,00/4,00 trở lên [quy-che-dao-tao-he-chinh-quy.md].
- Context kỳ vọng: data/standardized/legal/quy-che-dao-tao-he-chinh-quy.md, Điều 28 Công nhận tốt nghiệp
- Metrics: faithfulness=0.7083, relevance=0.6000, context_recall=1.0000, context_precision=1.0000

### 5. Học phí niêm yết theo tín chỉ là bao nhiêu?

- Điểm trung bình: 0.8872
- Câu trả lời kỳ vọng: Học phí niêm yết theo tín chỉ là 27.195.000 đồng cho mỗi tín chỉ.
- Câu trả lời thực tế: Học phí niêm yết theo tín chỉ là 27.195.000 đồng/tín chỉ. Mức học phí này được áp dụng trong các trường hợp như sinh viên học bổ sung tín chỉ khi học cùng lúc hai chương trình (double degree) tại VinUni, hoặc học chương trình tích hợp trao đổi tại các trường đối tác, và một số trường hợp khác [Document 1 \| Source: quy-dinh-tai-chinh.md].
- Context kỳ vọng: data/standardized/legal/quy-dinh-tai-chinh.md, mục Học phí niêm yết theo tín chỉ
- Metrics: faithfulness=0.8704, relevance=0.7500, context_recall=0.9286, context_precision=1.0000

## Đề xuất cải tiến

- Tất cả metric tổng hợp đều đạt ngưỡng 0.7; ưu tiên cải thiện các trường hợp điểm thấp nêu trên.
- Ưu tiên `hybrid_with_reranking` vì có điểm trung bình cao nhất; đồng thời cần cân nhắc latency trước khi triển khai.
- Kiểm tra refusal logic của generator cho hai trường hợp: relevant evidence đã được retrieve (Context Recall trên 0.81) nhưng model vẫn từ chối trả lời. Cần giữ section heading khi chunking và điều chỉnh prompt để phân biệt thiếu evidence với retrieval score thấp.
- Kiểm tra thủ công yêu cầu số tín chỉ tối thiểu vì generated answer là 12 tín chỉ còn golden answer là 14. Chỉnh golden dataset hoặc source data trước khi xem đây là lỗi của model.
- Cải thiện Answer Relevance bằng cách yêu cầu generator trả lời trực tiếp trước khi bổ sung chi tiết; Relevance (0.7363) là metric tổng hợp thấp nhất.
