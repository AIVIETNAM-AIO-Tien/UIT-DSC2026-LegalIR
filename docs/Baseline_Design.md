# LegalIR Baseline Design

> **Project:** UIT DSC 2026 — Legal Information Retrieval
> **Version:** Baseline v0.1

---

## 1. Objective

Baseline bài toán **Legal Information Retrieval (LegalIR)** của cuộc thi UIT DSC 2026.

Mục tiêu của Baseline v0.1:

1. Thiết lập benchmark ban đầu cho Candidate Retrieval.
2. So sánh hai hướng Retrieval cơ bản chp mục tiêu tối đa khả năng Candidate Retrieval:
   * BM25 — Lexical Retrieval
   * Dense Retrieval — Embedding-based Retrieval

---

# 2. Problem Formulation

Bài toán:

```text
Question
    +
Legal Document Corpus
        │
        ▼
Candidate Retrieval
        │
        ▼
Ranked Documents
        │
        ▼
Recall@K
```

Với mỗi query:

```text
q = user question

D = legal document corpus

GT(q) = set of relevant documents
```

Retrieval system tạo ra:

```text
R_K(q) = top-K retrieved documents
```

Mục tiêu của Candidate Retrieval là tối đa hóa khả năng các document thuộc `GT(q)` xuất hiện trong `R_K(q)`.

Metric chính của baseline:

```text
Recall@K
Recall@10
Recall@50
Recall@100
```

---

# 3. Baseline Philosophy

Baseline v0.1 tuân theo ba nguyên tắc:

### 3.1 Reproducibility

Một experiment phải có:

```text
Configuration
+
Code version
+
Dataset
+
Result
```

để có thể chạy lại và kiểm tra.

### 3.2 Separation of Components

Các thành phần được tách biệt:

```text
Data
  ↓
Preprocessing
  ↓
Chunking
  ↓
Retrieval
  ↓
Document Ranking
  ↓
Evaluation
```

Một Retrieval method không tự thực hiện toàn bộ pipeline.

### 3.3 Benchmark Before Optimization

Trước khi tối ưu một thành phần, phải có benchmark hiện tại để trả lời:

> Thay đổi này có thực sự cải thiện Candidate Recall hay không?

---

# 4. Overall Pipeline

Baseline v0.1 gồm pipeline chung:

```text
                         Legal Corpus
                              │
                              ▼
                       Data Loading
                              │
                              ▼
                        Normalization
                              │
                              ▼
                         Tokenization
                              │
                              ▼
                       Fixed Chunking
                              │
                              ▼
                         Chunk Corpus
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
                  BM25                Dense
                    │                   │
                    ▼                   ▼
               Chunk Scores        Chunk Scores
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
                    Document Aggregation
                              │
                              ▼
                      Document Ranking
                              │
                              ▼
                           Top-K
                              │
                              ▼
                         Recall@K
```

BM25 và Dense là **hai retrieval baseline độc lập**.

---

# 5. Data Layer

## 5.1 Legal Document Corpus

Corpus được lấy từ:

```text
selected-contexts/
```

Các file có dạng:

```text
context_<document_id>.json
```

Document được biểu diễn tối thiểu bằng:

```text
document_id
text
metadata
```

`document_id` là định danh chính của document và phải được giữ nguyên trong toàn bộ pipeline.

---

## 5.2 Query Dataset

Query và Ground Truth được lấy từ dataset của cuộc thi.

Mỗi query tối thiểu có:

```text
query_id
question
answers
```

Trong đó:

```text
question
```

là truy vấn của người dùng và:

```text
answers
```

là danh sách document ID được xem là Ground Truth.

Một query có thể có nhiều Ground Truth documents.

---

# 6. Preprocessing

Baseline sử dụng preprocessing chung cho corpus và query.

Pipeline:

```text
Raw Text
   │
   ▼
Normalization
   │
   ▼
Vietnamese Tokenization
```

## 6.1 Normalization

Mục tiêu là đưa text về representation thống nhất trước khi Retrieval.

Baseline sử dụng:

```text
normalize_text()
```

Chi tiết normalization phải được ghi nhận trong implementation và experiment configuration.

---

## 6.2 Vietnamese Tokenization

Baseline BM25 sử dụng:

```text
PyVi
```

Pipeline:

```text
Text
 ↓
PyVi
 ↓
Tokens
```

Query và corpus phải sử dụng preprocessing tương thích.

---

## 6.3 Stopword Removal

Baseline BM25-v0 hiện tại chưa có Stopword Removal

Cần thử stopword removal:

```text
BM25-v0
vs
BM25 + Stopword Removal
```

và được đánh giá bằng Recall@K.

---

# 7. Chunking

Baseline v0.1 sử dụng:

```text
Fixed-size Chunking
```

Configuration hiện tại:

```text
CHUNK_SIZE = 2048 tokens
OVERLAP = 256 tokens
```

Pipeline:

```text
Document
   │
   ▼
Fixed-size Chunking
   │
   ▼
Chunk 0
Chunk 1
Chunk 2
...
```

Mỗi chunk phải giữ được quan hệ với document gốc:

```text
chunk_id
document_id
chunk_index
text
```

---


## 7.2 Chunking by Legal Structure Chunking

Fixed-size chunking chỉ là baseline.

Các hướng cần thử sau:

```text
Fixed-size
    ↓
Legal Structure Chunking
    ↓
Article
    ↓
Clause
    ↓
Point
```

So sánh và đánh giá bằng Recall@K.

---

# 8. BM25 Baseline

## 8.1 Pipeline

BM25 baseline hiện tại:

```text
Processed Documents
        │
        ▼
Fixed-size Chunks
        │
        ▼
BM25 Index
        │
        ▼
Query
        │
        ▼
PyVi Tokenization
        │
        ▼
Query Tokens
        │
        ▼
BM25 Scores
        │
        ▼
Chunk Ranking
        │
        ▼
MAX Score per Document
        │
        ▼
Document Ranking
        │
        ▼
Top-100
        │
        ▼
Recall@100
```

---

## 8.2 Document Aggregation

BM25 trả về score ở mức chunk.

Do Ground Truth ở mức document, các chunk thuộc cùng một document được gom lại.

Baseline sử dụng:

```text
MAX score per document
```

Công thức:

[
Score(d,q)
==========

\max_{c \in C_d}
Score(c,q)
]

Trong đó:

* `d` là document;
* `C_d` là tập chunk thuộc document `d`;
* `Score(c,q)` là BM25 score giữa query và chunk.

Sau aggregation:

```text
Chunk Scores
     ↓
Document Scores
     ↓
Document Ranking
```

---

## 8.3 Current BM25 Benchmark

Kết quả hiện tại:

```text
BM25 Recall@100 = 0.9458
```

Các query không đạt Recall@100 = 1 được lưu để phân tích failure:

```text
bm25_fail_queries.json
```

Đây là benchmark chính thức của BM25 Baseline v0.1.

---

# 9. Dense Retrieval Baseline

Dense Retrieval sử dụng embedding model:

```text
Vietnamese_Embedding_v2
```

## 9.1 Corpus Encoding

```text
Chunk Corpus
     │
     ▼
Normalized Chunk Text
     │
     ▼
Vietnamese_Embedding_v2
     │
     ▼
Chunk Embeddings
```

## 9.2 Query Encoding

```text
Query
  │
  ▼
Normalized Query
  │
  ▼
Vietnamese_Embedding_v2
  │
  ▼
Query Embedding
```

## 9.3 Retrieval

```text
Query Embedding
       +
Chunk Embeddings
       │
       ▼
Similarity Score
       │
       ▼
Chunk Ranking
       │
       ▼
MAX Score per Document
       │
       ▼
Document Ranking
       │
       ▼
Top-100
       │
       ▼
Recall@100
```

Similarity hiện tại sử dụng:

```text
Dot Product
```

Dense baseline đang trong quá trình chạy benchmark.

Do đó:

```text
Dense Recall@100 = 0.9840
```

Không được ghi nhận một giá trị Dense benchmark trước khi experiment thực sự hoàn thành.

---

# 10. Common Document Ranking

BM25 và Dense đều hoạt động ở mức chunk:

```text
Query
  ↓
Chunk Scores
```

Trong khi Ground Truth ở mức document.

Vì vậy cả hai sử dụng cùng một bước:

```text
Chunk Scores
     ↓
MAX per Document
     ↓
Document Scores
     ↓
Top-K Documents
```

Điều này giúp benchmark giữa BM25 và Dense có cùng logic document-level ranking.

---

# 11. Evaluation

Evaluation nhận:

```text
Predicted Document IDs
+
Ground Truth Document IDs
```

và tính:

```text
Recall@K
```

Các K được theo dõi:

```text
Recall@10
Recall@20
Recall@50
Recall@100
```

Metric quan trọng nhất trong Baseline:

```text
Recall@100
```

---

## 11.1 Why Recall@100

Mục tiêu của Candidate Retrieval là đưa Ground Truth vào candidate set.

Nếu:

```text
GT ∉ Candidate Set
```

thì các bước ranking phía sau không thể khôi phục document đó.

Do đó Baseline ưu tiên:

```text
Candidate Recall
```

trước khi nghiên cứu các bước tối ưu Precision hoặc Reranking.

---

# 12. Baseline Benchmark Table

Benchmark được quản lý tập trung:

| Method   | Preprocessing        | Chunking       | Similarity  | Aggregation | Recall@100 |
| -------- | -------------------- | -------------- | ----------- | ----------- | ---------: |
| BM25-v0  | Normalization + PyVi | Fixed 2048/256 | BM25        | MAX         | **0.9458** |
| Dense-v0 | Normalization        | Fixed          | Dot Product | MAX         |    **0.9840** |

Khi một phương pháp mới được thử nghiệm, không ghi đè kết quả cũ.

Thêm một experiment/version mới.

---

# 13. What Belongs to Baseline v0.1

### Included

```text
✓ Dataset Loading
✓ Text Normalization
✓ PyVi Tokenization
✓ Fixed-size Chunking
✓ BM25
✓ Dense Retrieval
✓ Dot Product
✓ MAX per Document
✓ Document Ranking
✓ Recall@K
✓ Experiment Result Tracking
```

### Not Included

```text
✗ Hybrid Retrieval
✗ RRF
✗ Reranker
✗ Query Expansion
✗ HyDE
✗ LLM-based Retrieval
✗ Advanced Legal Chunking
✗ Advanced Query Understanding
```

Các thành phần này có thể được phát triển sau khi baseline hoàn thành.

---

# 14. Extension Points

Kiến trúc baseline được thiết kế để có thể mở rộng theo từng chiều.

## 14.1 Preprocessing

```text
Current:
Normalization + PyVi

Future:
Normalization variants
Tokenizer variants
Stopword experiments
```

## 14.2 Chunking

```text
Current:
Fixed-size 2048/256

Future:
Different chunk sizes
Legal Structure Chunking
Article-level
Clause-level
Hybrid chunking
```

## 14.3 Retrieval

```text
Current:
BM25
Dense

Future:
Improved BM25
Improved Dense
Hybrid Retrieval
```

## 14.4 Ranking

```text
Current:
MAX per Document

Future:
Top-N aggregation
Average
Weighted aggregation
RRF
Reranking
```

---

# 15. Experiment Principle

Mỗi experiment phải thay đổi một hoặc một nhóm yếu tố có chủ đích.

Ví dụ:

```text
Experiment A:
Chunk size 2048 → 1024
```

Không đồng thời thay đổi:

```text
Tokenizer
+
Chunking
+
Retriever
+
Aggregation
```

nếu mục tiêu là hiểu tác động của một thành phần cụ thể.

Mỗi experiment nên ghi:

```text
Experiment ID
Configuration
Changes
Baseline
Result
Difference
Conclusion
```

---

# 16. Team Development Rules

Mỗi thành viên có thể tạo feature branch:

```text
feature/bm25-improvement
feature/dense-improvement
feature/legal-chunking
feature/evaluation
```

Nhưng mọi implementation phải tuân theo pipeline chung:

```text
Data
 ↓
Preprocessing
 ↓
Chunking
 ↓
Retrieval
 ↓
Document Ranking
 ↓
Evaluation
```

Không tạo một pipeline riêng hoàn toàn độc lập nếu thay đổi đó có thể tích hợp vào pipeline chung.

---

# 17. Baseline Freeze

BM25 baseline được xác nhận:

```text
BM25-v0.1
Recall@100 = 0.9458
```

phiên bản này được freeze.

Có thể đánh dấu bằng Git tag:

```text
baseline-bm25-v0.1
```

Dense sẽ được freeze sau khi benchmark hoàn thành:

```text
baseline-dense-v0.1
```

Baseline được giữ nguyên để tất cả experiment sau này có thể so sánh.

---

# 18. Development Loop

Quá trình phát triển xuyên suốt cuộc thi:

```text
                 Baseline
                    │
                    ▼
                Experiment
                    │
                    ▼
                 Evaluate
                    │
             ┌──────┴──────┐
             │             │
           Worse         Better
             │             │
             ▼             ▼
          Discard        Analyze
                           │
                           ▼
                         Merge
                           │
                           ▼
                     New Benchmark
                           │
                           ▼
                       Experiment
                           │
                          ...
```

Mục tiêu là xây dựng một quy trình:

```text
Hypothesis
    ↓
Implementation
    ↓
Experiment
    ↓
Benchmark
    ↓
Analysis
    ↓
Improvement
```

---

# 19. Current Status

Tại thời điểm tạo Baseline v0.1:

```text
Dataset Understanding       ✓
Dataset Profiling            ✓
BM25 Baseline                ✓
BM25 Recall@100              0.9458
BM25 Failure Analysis        ✓
Dense Retrieval              In Progress
Dense Recall@100             TBD
Hybrid Retrieval             Not Started
Legal Structure Chunking     Not Started
Reranking                    Not Started
```

---

# 20. Definition of Done — Baseline v0.1

Baseline v0.1 được xem là hoàn thành khi:

```text
[ ] Dataset loader hoạt động
[ ] Common preprocessing hoạt động
[ ] Fixed-size chunking hoạt động
[ ] BM25 pipeline hoạt động
[ ] BM25 Recall@100 ≈ 0.9458 được reproduce
[ ] BM25 failure queries được lưu
[ ] Dense pipeline hoạt động
[ ] Dense Recall@100 được benchmark
[ ] Common document aggregation được sử dụng
[ ] Recall@K evaluator hoạt động
[ ] Configuration được ghi nhận
[ ] README có hướng dẫn chạy
[ ] Baseline được Git tag
```

Sau khi đạt các điều kiện trên, repository chuyển từ:

```text
Baseline Construction
```

sang:

```text
Retrieval Optimization
```

và team có thể bắt đầu phát triển các hướng BM25 improvement, Dense improvement, Legal Structure Chunking và Hybrid Retrieval.
