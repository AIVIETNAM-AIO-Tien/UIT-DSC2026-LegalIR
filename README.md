# ARCHITECTURE & LOGIC OVERVIEW: IT-DSC2026_LEGALIR

Baseline triển khai **Hệ thống Truy xuất Thông tin (Information Retrieval System)** đa phương thức. Hệ thống kết hợp giữa Dense Retrieval và Lexical BM25 Retrieval.

---

## 1. Core Architecture

Hệ thống chia làm 3 tầng chức năng chính:

```text
[ Dữ liệu Văn bản (Documents) ]
              │
              ├───> 1. Dense Retrieval Pipeline
              │      ├── Chunking (Tokenizer Windowing)
              │      ├── Embedding Generation (Transformer Model)
              │      └── Vector Search (Dot Product Index)
              │
              ├───> 2. Lexical Retrieval Pipeline
              │      ├── Multiprocessing Tokenization
              │      ├── Core Algorithms (src/retrieval/rank_bm25.py) -> BM25 Variants (Okapi, BM25L, BM25Plus)
              │      └── High-level Retriever (src/retrieval/bm25.py)
              │
              ▼
[ Kết quả Truy xuất (Retrieval Results) ]
              │
              ▼
    3. Submission Builder
              │
              ▼
     [ submission.json ]

```

---

## 2. Detailed Components

### Module: Tiền xử lý Dữ liệu (Text Preprocessing)

*File liên quan: `src/preprocessing/normalize.py` & `src/preprocessing/tokenize.py*`

Module tiền xử lý đảm nhận nhiệm vụ làm sạch, chuẩn hóa định dạng và phân tách từ tiếng Việt cho toàn bộ văn bản thô (cả đoạn văn bản lẫn câu hỏi) trước khi đưa vào các pipeline truy xuất.

---

#### 1. Chuẩn hóa Văn bản (`src/preprocessing/normalize.py`)

Hàm `normalize_text` thực hiện các bước làm sạch cơ bản nhằm đảm bảo tính đồng nhất của dữ liệu mà không làm mất mát thông tin ngữ nghĩa (Triết lý thiết kế: *Đồng nhất dữ liệu - không cắt giảm thông tin*).

* **Kiểm tra Kiểu dữ liệu:** Đảm bảo đầu vào luôn là một chuỗi (`str`), ném lỗi `TypeError` nếu sai kiểu dữ liệu.
* **Chuẩn hóa Unicode (NFC):** Chuyển đổi toàn bộ ký tự tiếng Việt về chuẩn **Unicode NFC (Normalization Form C)**. Điều này giải quyết triệt để lỗi sai lệch mã giữa dấu tổ hợp và dấu dựng sẵn thường gặp trong tiếng Việt.
* **Đồng nhất Xuống dòng:** Chuyển tất cả các định dạng xuống dòng (`\r\n`, `\r`) về dạng chuẩn `\n`.
* **Gộp Khoảng trắng (Whitespace Collapsing):** Chuyển đổi các chuỗi khoảng trắng liên tiếp, ký tự tab hoặc dòng trống thừa (`\s+`) thành 1 khoảng trắng duy nhất.
* **Xóa Khoảng trắng Thừa:** Loại bỏ khoảng trắng ở hai đầu văn bản bằng `.strip()`.

---

#### 2. Tách từ Tiếng Việt (`src/preprocessing/tokenize.py`)

Hàm `tokenize_vietnamese` đóng vai trò là bộ tách từ tiếng Việt (Word Tokenizer), tạo tiền đề cho công đoạn lập chỉ mục từ khóa của BM25.

* **Kiểm tra Kiểu dữ liệu:** Đảm bảo đầu vào là chuỗi đã qua bước chuẩn hóa (`normalize_text`).
* **Phân tách Từ ghép (Vietnamese Word Segmentation):** Sử dụng thư viện `PyVi` (`ViTokenizer.tokenize`) để nhận diện và nối các từ ghép tiếng Việt bằng dấu gạch dưới (ví dụ: `"ngôn ngữ lập trình"` $\rightarrow$ `"ngôn_ngữ lập_trình"`).
* **Định dạng Đầu ra:** Chuyển đổi chuỗi đã tách từ thành danh sách các token (`list[str]`) thông qua phương thức `.split()`.
* **Tích hợp:** Hàm này chính là hàm callback được truyền vào tham số `tokenize_fn` khi khởi tạo `BM25Retriever` (`src/retrieval/bm25.py`).

---

#### 3. Vai trò trong Data Pipeline Overall

```text
[ Raw Text / Query ] 
         │
         ▼
`normalize_text()` ──(Chuẩn hóa NFC, khoảng trắng)──> [ Cleaned Text ]
                                                             │
                                   ┌─────────────────────────┴─────────────────────────┐
                                   ▼                                                   ▼
                         Dense Retrieval Pipeline                   `tokenize_vietnamese()`
                    (Dùng trực tiếp Tokenizer của Transformer)                          │
                                                                                        ▼
                                                                             [ List of Tokens ]
                                                                                        │
                                                                                        ▼
                                                                              BM25Retriever (`fit`/`retrieve`)

```
---

### Module 1: (Dense Retrieval)

*File chính: `src/retrieval/dense.py*`

Nhiệm vụ của module này là hiểu ngữ nghĩa của câu hỏi và đoạn văn bản, cho phép tìm kiếm ngay cả khi câu hỏi và văn bản không dùng chung từ khóa exact-match.

* **Cơ chế Cắt đoạn (Chunking Mechanism):**
* Chia nhỏ các tài liệu văn bản dài thành nhiều đoạn nhỏ (**Chunks**).
* Sử dụng kỹ thuật **Sliding Window** dựa trên Tokenizer:
* Kích thước đoạn (`chunk_size`): 2048 tokens.
* Độ gối đầu (`overlap`): 256 tokens.




* **Mô hình Embedding (Vector Representation):**
* Sử dụng mô hình tiền huấn luyện chuyên biệt cho tiếng Việt: `AITeamVN/Vietnamese_Embedding_v2`.
* Chuyển đổi từng Chunk văn bản thành một vector đa chiều đại diện cho ngữ nghĩa.


* **Lưu trữ & Khôi phục Index (Persistence):**
* Lưu trữ cấu trúc metadata của Chunk (ID tài liệu, vị trí token bắt đầu/ket thúc, đoạn text) dưới dạng file JSON.
* Lưu trữ toàn bộ Ma trận Vector dưới dạng file nhị phân NumPy (`.npy`).
* Cho phép nạp lại Index nhanh chóng mà không cần tốn chi phí tính toán Embeddings lại từ đầu.


* **Logic Tìm kiếm (Vector Search Strategy):**
* Khi có câu hỏi (Query), câu hỏi sẽ được mã hóa thành 1 Query Vector.
* Tính điểm tương đồng bằng **Tích vô hướng (Dot Product)** giữa Query Vector và toàn bộ Chunk Embeddings.
* Lọc và sắp xếp lấy ra **Top-K** đoạn văn bản có độ tương đồng ngữ nghĩa cao nhất.



---

### Module 2: Truy xuất Từ khóa (Lexical BM25 Retrieval)

*File liên quan: `src/retrieval/bm25.py` (Quản lý quy trình) & `src/retrieval/rank_bm25.py` (Thuật toán cốt lõi)*

Module này đảm nhận nhiệm vụ tìm kiếm chính xác dựa trên tần suất xuất hiện của từ khóa (Exact Keyword Match), phục vụ tốt cho các trường hợp tìm kiếm tên riêng, mã số, hoặc thuật ngữ chuyên ngành. Kiến trúc module được chia làm 2 tầng rõ ràng:

#### 1. Tầng Thuật toán & Chỉ mục Cốt lõi (`src/retrieval/rank_bm25.py`)

* Đóng vai trò là thư viện tính toán nền tảng.
* Quản lý **Chỉ mục Ngược (Inverted Index)**: Thống kê tần suất từ trong đoạn (TF), số lượng đoạn chứa từ (DF), độ dài từng đoạn văn và độ dài trung bình toàn tập (`avgdl`).
* Tối ưu hóa tốc độ tiền xử lý bằng cơ chế tách từ song song trên đa nhân CPU (`multiprocessing.Pool`).
* Cung cấp các công thức tính điểm toán học cho 3 biến thể BM25 (`BM25Okapi`, `BM25L`, `BM25Plus`).

---

#### 2. Tầng Bộ truy xuất Cấp cao & Interface (`src/retrieval/bm25.py` - Chi tiết Logic)

Lớp `BM25Retriever` kế thừa từ giao diện chung `Retriever`, chịu trách nhiệm kết nối dữ liệu thô (Schema `Chunk`, `Query`) với thuật toán `BM25Okapi` để thực hiện tìm kiếm ở cấp độ Chunk (Chunk-level Retrieval).

* **Khởi tạo (`__init__`):**
* Tiếp nhận một hàm tách từ linh hoạt (`tokenize_fn: Callable[[str], list[str]]`). Điều này giúp người dùng dễ dàng thử nghiệm các công cụ tách từ tiếng Việt khác nhau (như ViTokenizer, PyVi, Underthesea, v.v.).
* Thiết lập hai siêu tham số BM25 mặc định:
* `k1 = 1.5`: Bão hòa tần suất từ (Term frequency saturation).
* `b = 0.75`: Mức độ phạt độ dài tài liệu (Document length penalty).


* Khai báo bộ nhớ đệm nội bộ: `_bm25` (Lưu instance BM25) và `_chunks` (Lưu danh sách các Chunk gốc).


* **Lập chỉ mục (`fit`):**
* **Kiểm tra đầu vào:** Đảm bảo danh sách `chunks` truyền vào không bị rỗng (`ValueError`).
* **Tách từ toàn bộ Corpus:** Duyệt qua danh sách `Chunk` và dùng `tokenize_fn` để tách nội dung văn bản (`chunk.text`) thành danh sách các token: `tokenized_corpus`.
* **Khởi tạo BM25 Index:** Khởi tạo instance `BM25Okapi` từ `tokenized_corpus` cùng các tham số `k1`, `b`.


* **Logic Truy xuất (`retrieve`):**
* **Kiểm tra ràng buộc:** Đảm bảo `top_k > 0`. Nếu danh sách `chunks` rỗng, lập tức trả về danh sách kết quả trống `[]`.
* **Cơ chế Lazily Build Index:** Nếu `fit()` chưa từng được gọi thủ công trước đó (`self._bm25 is None`), hệ thống sẽ tự động gọi `fit(chunks)` để tạo Index ngay lập tức.
* **Mã hóa Truy vấn (Query Tokenization):** Chuyển đổi câu hỏi `query.question` thành danh sách các từ/token bằng `tokenize_fn`.
* **Tính điểm Tương đồng (Scoring):** Chuyển câu hỏi đã tách từ vào `self._bm25.get_scores()`, trả về một mảng chứa điểm số BM25 cho tất cả các Chunks trong tập dữ liệu.
* **Sắp xếp & Cắt Top-K (Ranking & Truncation):**
* Sử dụng mảng chỉ số (indices) để sắp xếp giảm dần theo điểm số (`reverse=True`).
* Cắt lấy danh sách `top_k` đoạn văn bản có điểm cao nhất.


* **Đóng gói Kết quả (Formatting Output):** Duyệt qua các vị trí đã sắp xếp, tính toán thứ hạng (`rank` bắt đầu từ 1) và khởi tạo đối tượng chuẩn `RetrievalResult` chứa các thông tin:
* `chunk_id`: Định danh duy nhất của đoạn văn.
* `document_id`: Định danh của tài liệu gốc chứa đoạn văn đó.
* `score`: Điểm tương đồng BM25 (`float`).
* `rank`: Thứ hạng tương ứng (1, 2, 3...).



---


### Module 3: Đóng gói Kết quả Đầu ra (Submission Builder)

*File chính: `src/submission/builder.py*`

Nhiệm vụ của module này là tiếp nhận danh sách tài liệu đã được rank/sắp xếp theo độ liên quan từ các bước truy xuất và chuyển đổi thành định dạng nộp bài tiêu chuẩn.

* **Kiểm tra Ràng buộc (Validation Rules):**
* Đảm bảo số lượng tài liệu trả về cho mỗi câu hỏi nằm trong giới hạn cho phép ($k \le 5$).


* **Logic Ánh xạ (Mapping Logic):**
* Chuyển đổi định dạng từ `question_id -> [list_document_ids]` sang cấu trúc JSON chuẩn:
```json
{
  "question_id_1": {
    "answer": ["doc_id_1", "doc_id_2", "doc_id_3", "doc_id_4", "doc_id_5"]
  }
}

```




* **Xuất dữ liệu:** Đảm bảo mã hóa UTF-8 để giữ nguyên ký tự tiếng Việt và ghi dữ liệu ra file `submission.json`.

---

## 3. Luồng dữ liệu Xử lý End-to-End (Data Pipeline Flow)

1. **Giai đoạn Offline (Index Construction):**
* Dense: Tập văn bản thô $\rightarrow$ Phân đoạn (Chunking) $\rightarrow$ Sinh Embeddings $\rightarrow$ Lưu Index (`dense_chunks.json`, `dense_chunk_embeddings.npy`).
* BM25:  Tập văn bản thô $\rightarrow$ Khởi tạo BM25Retriever $\rightarrow$ Tách từ bằng tokenize_fn $\rightarrow$ Xây dựng chỉ mục qua fit() bằng BM25Okapi.


2. **Giai đoạn Online (Inference / Retrieval):**
* Nhận Query từ người dùng.
* Tính toán điểm số tương đồng ngữ nghĩa bằng Dense Retrieval và/hoặc tần suất từ khóa bằng BM25.
* Lấy danh sách Top các tài liệu có điểm số cao nhất.


3. **Giai đoạn Output Generation:**
* Cắt lấy Top-5 tài liệu liên quan nhất cho mỗi câu hỏi.
* Ghi kết quả ra file `submission.json` để hoàn thành quy trình.



---

## 4. Điểm mạnh trong Thiết kế Hệ thống

| Đặc điểm | Mô tả & Lợi ích |
| --- | --- |
| **Modular Scalability** | Các module Dense, BM25 và Submission tách biệt hoàn toàn, dễ dàng thay thế mô hình Embedding mới hoặc đổi thuật toán BM25 mà không ảnh hưởng toàn bộ hệ thống. |
| **Efficient Indexing** | Khả năng Save/Load index nhị phân giúp tiết kiệm thời gian chạy thử nghiệm và triển khai thực tế. |
| **Hybrid Potential** | Kiến trúc sẵn sàng cho việc kết hợp (Hybrid Search / Ensembling) giữa kết quả của Dense Retriever và BM25 Retriever thông qua các phương pháp RRF (Reciprocal Rank Fusion) hoặc Re-ranking. |

## RUN
> top-k: là top-k document không phải top-k chunks

* `run_dense_submission.py`
```bash
!PYTHONPATH=. python scripts/run_dense_submission.py \
  --test /content/drive/MyDrive/LEGALIR-DSC/public-official.json \
  --chunks /content/drive/MyDrive/LEGALIR-DSC/output/dense/dense_chunks.json \
  --embeddings /content/drive/MyDrive/LEGALIR-DSC/output/dense/dense_chunk_embeddings.npy \
  --output /content/drive/MyDrive/LEGALIR-DSC/output/submission.json \
  --top-k 5
```

* `run_dense.py`
  * load index
```bash
!PYTHONPATH=. python scripts/run_dense.py \
    --train /content/drive/MyDrive/LEGALIR-DSC/train.json \
    --chunks /content/drive/MyDrive/LEGALIR-DSC/output/dense/dense_chunks.json \
    --embeddings /content/drive/MyDrive/LEGALIR-DSC/output/dense/dense_chunk_embeddings.npy \
    --output /content/drive/MyDrive/LEGALIR-DSC/output/dense/dense_result.json \
    --top-k 100
```

  * build index

```bash
!PYTHONPATH=. python /content/UIT-DSC2026-LegalIR/scripts/run_dense.py \
    --train /content/drive/MyDrive/LEGALIR-DSC/train.json \
    --contexts /content/drive/MyDrive/LEGALIR-DSC/selected-contexts \
    --output /content/drive/MyDrive/LEGALIR-DSC/output/dense/ense_result.json \
    --output-chunks /content/drive/MyDrive/LEGALIR-DSC/output/dense/dense_chunks.json \
    --output-embeddings /content/drive/MyDrive/LEGALIR-DSC/output/dense/dense_chunk_embeddings.npy \
    --top-k 100
```