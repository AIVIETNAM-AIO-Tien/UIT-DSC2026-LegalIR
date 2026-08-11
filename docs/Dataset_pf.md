# DataSet Profile
---

# 1. Cấu trúc dữ liệu
| Tệp | Nội dung |
|------|----------|
| `train.json` | Tập dữ liệu huấn luyện cho các đội phát triển phương pháp. |
| `warmup.json` | Tập dữ liệu mẫu phục vụ vòng Warm-up, giúp làm quen bài toán và quy trình submission. |
| `public-official.json` | Tập dữ liệu dùng trong giai đoạn Public Test theo cấu hình chính thức. |
| `private-official.json` | Tập dữ liệu chính thức của Private Test. |
| `selected-contexts.zip` | Kho văn bản được chọn; gồm nhiều tệp `context_*.json`. |

## 1.1 `selected-contexts.zip`
* Tên file `context_id.json`, id không liên tục, filename == document_id:
```text
context_21.json
context_69.json
```
## 1.2 `context_*.json` có dạng:
* id: mã định danh duy nhất của văn bản
* name: tiêu đề văn bản
* link: đường dẫn nguồn
* passage: toàn bộ nội dung văn bản trên Thư Viện Pháp Luật được sử dụng để truy vấn (có thể chứa noise). Bao gồm:
* Các passage trong các context_id.json có độ dài rất khác nhau.

`context_740.json` minh họa:
```json
{
    "link": "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608.aspx",
    "name": "Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608",
    "passage": "BỘ Y TẾ\r\n\n  -------\n\nCỘNG HÒA XÃ HỘI\r\n\n  CHỦ NGHĨA VIỆT NAM\r\n\n  Độc lập - Tự do - Hạnh phúc \r\n\n  ---------------\n\nSố: 5868/QĐ-BYT\n\nHà Nội, ngày 28\r\n\n  tháng 9 năm 2018\n\n\n\nQUYẾT ĐỊNH\n\nQUY\r\n\nĐỊNH CHỨC NĂNG, NHIỆM VỤ, QUYỀN HẠN VÀ CƠ CẤU TỔ CHỨC CỦA VỤ TRANG THIẾT BỊ VÀ CÔNG\r\n\nTRÌNH Y TẾ THUỘC BỘ Y TẾ\n\nBỘ TRƯỞNG BỘ Y TẾ\n\nCăn cứ Nghị định số 75/2017/NĐ-CP … - Lưu: VT, TCCB, TTB, PC.\n\nBỘ TRƯỞNG\n\n\r\n\n  Nguyễn Thị Kim Tiến\n\n",
    "id": 740
}
```
- Trường `name` vẫn là một nguồn metadata hữu ích, nhưng nếu muốn sử dụng cho các tác vụ nhận dạng hoặc phân loại trong các giai đoạn sau, cần có bước chuẩn hóa (normalize) riêng.

### 1.2.1 Length Statistics

```json
{
    "total_documents": 8532,
    "total_tokens": 89404593,
    "average_document_length": 10478.738045007032,
    "median_document_length": 5827.5,
    "min_document_length": 0,
    "max_document_length": 1690872,
    "std_document_length": 24896.511544402816,
    "p50": 5827.5,
    "p75": 11428.75,
    "p90": 22291.0,
    "p95": 33621.7,
    "p99": 72684.16000000003,
    "token_bucket(BAAI/bge-m3)": {
        "<512": 48,
        "512-1024": 257,
        "1024-2048": 852,
        "2048-4096": 1898,
        "4096-8192": 2371,
        ">8192": 3106
    }
}
```

### 1.2.2 metadata report
- missing_id : 0,
- missing_name : 1125,
- missing_link : 0,
- missing_passage : 20
- Toàn bộ Link đều có domain `thuvienphapluat.vn`
- Duplicate ID: 0
- Duplicate Name: 1125 == missing_name
- Duplicate Passage: 29
- invalid_url: 0

### 1.2.3 Legal Structure Statistics

| Structure |  Mean |    Std | Min |    Q1 | Median |     Q3 |   Max | Percentage |
| --------- | ----: | -----: | --: | ----: | -----: | -----: | ----: |
| Chương    |  2.83 |   4.13 |   0 |  0.00 |   3.00 |   5.00 |   133 |     51.78% |
| Mục       |  1.03 |   4.31 |   0 |  0.00 |   0.00 |   0.00 |   227 |     17.58% |
| Điều      | 19.17 |  30.47 |   0 |  4.00 |  12.00 |  25.00 |   838 |     85.90% |
| Khoản     | 97.45 | 210.17 |   0 | 22.00 |  49.00 | 104.00 | 7,565 |     97.26% |
| Điểm      | 53.06 | 103.96 |   0 |  4.00 |  22.00 |  59.00 | 2,936 |     80.38% |

### 1.2.4 Oversized Legal Units

| Legal Unit |   >512 |  >768 | >1024 | >1536 | >2048 |
| ---------- | -----: | ----: | ----: | ----: | ----: |
| Điều       | 17.46% | 9.30% | 5.89% | 3.26% | 2.33% |
| Khoản      |  2.26% | 1.10% | 0.65% | 0.30% | 0.18% |
| Điểm       |  5.03% | 3.12% | 2.14% | 1.21% | 0.81% |

### 1.2.5 Document Type Distribution

Document được phân loại dựa vào chủ yếu đoạn đầu của trường `passage`, có thể dùng thử trường `name` nhưng cần chuẩn hóa lại.

| Document Type | Count | Percentage |
| ------------- | ----: | ---------: |
| Luật          | 5,184 |     60.76% |
| Nghị định     | 1,491 |     17.48% |
| Khác          |   856 |     10.03% |
| Bộ luật       |   274 |      3.21% |
| Quyết định    |   254 |      2.98% |
| Thông tư      |   174 |      2.04% |
| Quy định      |    88 |      1.03% |
| Tiêu chuẩn    |    55 |      0.64% |
| Chỉ thị       |    50 |      0.59% |
| Quy chế       |    46 |      0.54% |
| Thông báo     |    37 |      0.43% |
| Công văn      |    13 |      0.15% |
| Quy chuẩn     |    10 |      0.12% |

### 1.2.6 Document Quality

Passage có thể chứa HTML/JavaScript và nội dung ngoài phần luật thực sự.
```text
- HTML / JavaScript
- Header / footer
- Quốc hiệu / tiêu ngữ
- Chữ ký
- Nơi nhận
- Ký tự bất thường
```

## 1.3 `train.json` và `public-official.json`

```json
{
    "19826": {
        "question": "Người có hành vi xúc phạm tôn giáo thì sẽ bị xử phạt như thế nào theo quy định pháp luật?",
        "answer": [
            "44802",
            "65293"
        ]
    },
    "88634": {
        "question": "Chứng chỉ hành nghề dược được quản lý như thế nào?",
        "answer": [
            "33079"
        ]
    }
}
```
> Ground Truth có thể gồm nhiều document.

```json
{
    "38096": {
        "question": "Đề nghị xem xét lại quyết định đình chỉ tiến hành thủ tục phá sản được xem xét, giải quyết trong thời hạn bao nhiêu ngày làm việc?",
        "answer": null
    },
    "63410": {
        "question": "Người giúp việc cho Giám đốc Nhà khách Dân tộc có trách nhiệm như thế nào?",
        "answer": null
    }
}
```
### 1.3.1 Query Length

| dataset   | metric       |   mean |   median |   std |   min |    q1 |   q3 |   max |
|:----------|:-------------|-------:|---------:|------:|------:|------:|-----:|------:|
| train     | char_length  |  88.62 |       86 | 30.27 |    12 | 67    |  108 |   233 |
| public    | char_length  |  87.17 |       84 | 29.76 |    14 | 65.75 |  106 |   240 |
| train     | word_length  |  19.78 |       19 |  6.65 |     4 | 15    |   24 |    50 |
| public    | word_length  |  19.47 |       19 |  6.48 |     4 | 15    |   24 |    52 |
| train     | token_length |  21.77 |       21 |  7.09 |     4 | 17    |   26 |    60 |
| public    | token_length |  21.46 |       21 |  6.88 |     5 | 17    |   26 |    56 |

> `train.json`

```json
{
  "total_queries": 7000,
  "empty_queries": 0,
  "unique_queries": 6985,
  "duplicate_queries": 15,
}
```
> `public-official.json`

```json
{
  "total_queries": 1000,
  "empty_queries": 0,
  "unique_queries": 1000,
  "duplicate_queries": 0,
}
```

### 1.3.2 Query Intent Analysis Train vs Public

| primary_intent   |   train_percentage |   public_percentage |
|:-----------------|-------------------:|--------------------:|
| Other            |              39.2  |                38   |
| Condition        |              13.6  |                16.4 |
| Procedure        |              11.21 |                10.7 |
| Organization     |               8.29 |                 8.2 |
| Authority        |               7.34 |                 5.8 |
| Responsibility   |               6.34 |                 5.2 |
| Time             |               6.09 |                 7.6 |
| Penalty          |               4    |                 3.7 |
| Definition       |               3.71 |                 4.1 |
| Document lookup  |               0.21 |                 0.3 |

### 1.3.3 Train Answer Analysis

```json
{
  "total_queries": 7000,
  "null_answers": 0,
  "queries_with_answer": 7000,
  "answer_count_per_query": {
    "min": 1,
    "max": 5,
    "mean": 1.09,
    "median": 1.0,
    "std": 0.33,
    "q1": 1.0,
    "q3": 1.0
  }
}
```

|   answer_count |   queries |   percentage |
|---------------:|----------:|-------------:|
|              1 |      6447 |        92.1  |
|              2 |       485 |         6.93 |
|              3 |        53 |         0.76 |
|              4 |        14 |         0.2  |
|              5 |         1 |         0.01 |

### 1.3.4 Document Frequency

Document frequency measures how many times each corpus document appears in the ground-truth `answer` of the training queries.

```json
{
  "unique_used_documents": 3105,
  "total_answer_document_links": 7637,
  "frequency_per_document": {
    "min": 1,
    "max": 109,
    "mean": 2.46,
    "median": 1.0,
    "std": 5.05,
    "q1": 1.0,
    "q3": 2.0
  }
}
```

### 1.3.5 Lexical Overlap Statistics


| metric          |      mean |    median |      std |   min |       q1 |        q3 |       max |
|:----------------|----------:|----------:|---------:|------:|---------:|----------:|----------:|
| max_gt_jaccard  |  0.026516 |  0.023547 | 0.014921 |     0 | 0.016148 |  0.033708 |  0.263158 |
| mean_gt_jaccard |  0.026075 |  0.023294 | 0.0145   |     0 | 0.016015 |  0.032987 |  0.263158 |
| max_gt_bm25     | 15.9814   | 14.1077   | 9.31032  |     0 | 9.45289  | 20.5272   | 80.2916   |
| mean_gt_bm25    | 15.8519   | 14.01     | 9.25914  |     0 | 9.36155  | 20.3351   | 80.2916   |