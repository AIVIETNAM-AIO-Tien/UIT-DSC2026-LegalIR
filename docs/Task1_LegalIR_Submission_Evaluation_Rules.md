# QUY ĐỊNH NỘP BÀI VÀ QUY TẮC ĐÁNH GIÁ
## Task 1: Legal Information Retrieval (LegalIR) - UIT DSC 2026

---

## 1. MỤC TIÊU VÀ CẤU TRÚC BÀI TOÁN

* **Mục tiêu:** Cho một câu hỏi pháp luật bằng tiếng Việt, hệ thống cần truy vấn và xác định danh sách các mã định danh văn bản hành chính/pháp lý (`document_id`) chứa thông tin cần thiết để trả lời câu hỏi đó.
* **Input:** Câu hỏi pháp luật tiếng Việt (`question`).
* **Output:** Danh sách các mã văn bản (`document_id`) liên quan, được sắp xếp theo thứ tự giảm dần về mức độ liên quan.

```json
"id": {
        "question": "Việc tổ chức vận động, tiếp nhận, sử dụng nguồn đóng góp tự nguyện được thực hiện dựa trên nguyên tắc nào?",
        "answer": [
            "177504"
        ]
    }

```
## 2. Cấu trúc dữ liệu
| Tệp | Nội dung |
|------|----------|
| `train.json` | Tập dữ liệu huấn luyện cho các đội phát triển phương pháp. |
| `warmup.json` | Tập dữ liệu mẫu phục vụ vòng Warm-up, giúp làm quen bài toán và quy trình submission. |
| `public-official.json` | Tập dữ liệu dùng trong giai đoạn Public Test theo cấu hình chính thức. |
| `private-official.json` | Tập dữ liệu chính thức của Private Test. |
| `selected-contexts.zip` | Kho văn bản được chọn; gồm nhiều tệp `context_*.json`. |

## 3. Cấu trúc một văn bản
### `train.json`
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

### `public-official.json`
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

### `selected-contexts.zip`
* Gồm 8532 file json
* Tên file `context_id.json`, id không liên tục, filename == document_id:
```text
context_21.json
context_69.json
```
### `context_*.json` có dạng:
* id: mã định danh duy nhất của văn bản
* name: tiêu đề văn bản
* link: đường dẫn nguồn
* passage: toàn bộ nội dung văn bản trên Thư Viện Pháp Luật được sử dụng để truy vấn. Bao gồm:
    * tiêu đề
    * căn cứ
    * các điều
    * nơi nhận
    * chữ ký
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

---

## 4. QUY ĐỊNH VỀ TỆP NỘP BÀI (SUBMISSION FORMAT)

### 4.1. Cấu trúc tệp đóng gói
Thí sinh cần nộp một tệp ZIP có tên chính xác là `submission.zip`. Bên trong tệp ZIP chỉ chứa duy nhất một tệp JSON:

```text
submission.zip
└── submission.json
```

### 4.2. Định dạng tệp `submission.json`
* Tệp `submission.json` phải là một đối tượng JSON (JSON Object) được mã hóa dưới dạng **UTF-8**.
* Mỗi khóa (**key**) trong đối tượng JSON là mã định danh câu hỏi (`question_id`).
* Giá trị (**value**) tương ứng là một đối tượng chứa thuộc tính `"answer"`. Thuộc tính này chứa một mảng/danh sách các chuỗi `document_id`.
* Các `document_id` trong mảng phải được **sắp xếp theo thứ tự giảm dần về mức độ liên quan** (văn bản được đánh giá liên quan nhất phải nằm ở vị trí đầu tiên).

### 4.3. Ví dụ mẫu tệp `submission.json` hợp lệ
```json
{
    "q_000001": {
        "answer": [
            "doc_123",
            "doc_456",
            "doc_789",
            "doc_321",
            "doc_654"
        ]
    },
    "q_000002": {
        "answer": [
            "doc_100",
            "doc_205",
            "doc_501"
        ]
    }
}
```

---

## 5. CÁC RÀNG BUỘC & LƯU Ý BẮT BUỘC

### 5.1. Ràng buộc về số lượng văn bản trả về (Quy định phạt từ BTC)
* **Giới hạn số lượng:** Đối với **mỗi câu hỏi**, danh sách `"answer"` chỉ được chứa **tối đa 5 `document_id`** ($1 \le \text{số lượng answer} \le 5$).
* **Mức xử phạt cực nghiêm ngặt:** Nếu tệp nộp bài có **bất kỳ** câu hỏi nào trả về **nhiều hơn 5 `document_id`**, hệ thống đánh giá sẽ ngay lập tức hủy kết quả bài nộp đó và tính **0 điểm** cho cả **Recall** lẫn **Precision**.

### 5.2. Yêu cầu tính hợp lệ của dữ liệu
* **Tính duy nhất của câu hỏi:** Mỗi `question_id` phải xuất hiện đúng một lần trong tệp `submission.json`.
* **Trùng lặp document:** Không được phép xuất hiện `document_id` trùng lặp trong cùng một danh sách `answer` của một câu hỏi.
* **Tồn tại dữ liệu:** Các `document_id` được dự đoán phải tồn tại trong tập dữ liệu (kho văn bản) do Ban Tổ chức cung cấp.

### 5.3. Ví dụ về các trường hợp KHÔNG hợp lệ
1. **Trùng lặp document_id:**
   ```json
   {
       "q_000001": {
           "answer": ["doc_123", "doc_123", "doc_456"]
       }
   }
   ```
2. **Thiếu trường `answer`:**
   ```json
   {
       "q_000001": {}
   }
   ```
3. **Trường `answer` không phải là mảng:**
   ```json
   {
       "q_000001": {
           "answer": "doc_123"
       }
   }
   ```
4. **Vượt quá số lượng answer cho phép (> 5 văn bản):**
   ```json
   {
       "q_000001": {
           "answer": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
       }
   }
   ``` *(Trường hợp này sẽ khiến toàn bộ bài nộp nhận điểm 0)*

---

## 6. QUY TẮC ĐÁNH GIÁ & PHƯƠNG PHÁP XẾP HẠNG (EVALUATION METRICS)

Kết quả thi đấu sẽ được tính toán tự động trên hệ thống chấm điểm dựa theo 2 độ đo:

| Độ đo | Vai trò | Mô tả ngắn |
| :--- | :--- | :--- |
| **Recall** | **Độ đo chính (Primary Metric)** | Đo tỷ lệ các văn bản đúng (Ground Truth) được hệ thống truy xuất thành công. |
| **Precision** | **Độ đo phụ (Secondary Metric)** | Đo tỷ lệ các văn bản mà hệ thống truy xuất là đúng. Dùng phân định khi bằng điểm Recall. |

### 6.1. Độ đo chính: Recall
Recall đo tỷ lệ các văn bản đúng được hệ thống truy xuất. Đối với từng câu hỏi $i$:

$$\text{Recall}_i = \frac{|\text{Relevant}_i \cap \text{Predicted}_i|}{|\text{Relevant}_i|}$$

*Trong đó:*
* $\text{Relevant}_i$: Tập hợp các mã văn bản đúng của câu hỏi $i$.
* $\text{Predicted}_i$: Tập hợp các mã văn bản do hệ thống dự đoán trả về cho câu hỏi $i$.

Điểm Recall chung cuộc của tệp nộp bài là trung bình cộng của Recall trên toàn bộ $N$ câu hỏi:

$$\text{Recall} = \frac{1}{N} \sum_{i=1}^{N} \text{Recall}_i$$

*Giá trị Recall nằm trong khoảng $[0, 1]$, giá trị càng cao thể hiện kết quả càng tốt.*

### 6.2. Độ đo phụ: Precision
Precision đo tỷ lệ chính xác của các văn bản do hệ thống trả về. Đối với từng câu hỏi $i$:

$$\text{Precision}_i = \frac{|\text{Relevant}_i \cap \text{Predicted}_i|}{|\text{Predicted}_i|}$$

*Trong đó:*
* Nếu hệ thống không trả về văn bản nào cho câu hỏi $i$ (mảng `answer` rỗng), $\text{Precision}_i = 0$.

Điểm Precision chung cuộc là trung bình cộng của Precision trên toàn bộ $N$ câu hỏi:

$$\text{Precision} = \frac{1}{N} \sum_{i=1}^{N} \text{Precision}_i$$

*Giá trị Precision nằm trong khoảng $[0, 1]$, giá trị càng cao thể hiện kết quả càng tốt.*

### 6.3. Quy tắc phân định thứ hạng
1. Bảng xếp hạng (**Scoreboard**) căn cứ theo điểm **Recall** từ cao xuống thấp.
2. Nếu hai hoặc nhiều đội có **cùng điểm Recall**, thứ hạng ưu tiên sẽ được quyết định dựa trên điểm **Precision** (đội có Precision cao hơn sẽ đứng trên).

# Nội dung bổ sung về Vòng Warm-up
## 1.Thời gian triển khai
* Bắt đầu: 00:00 GMT+7, ngày 01/08/2026.
* Kết thúc: 23:59 GMT+7, ngày 05/08/2026.
## 2. Mô tả & Mục đích
**Mục đích**: Giúp các đội thi làm quen với dữ liệu, xây dựng pipeline nộp bài, kiểm thử mô hình và làm quen với hệ thống chấm điểm / bảng xếp hạng (Leaderboard).

**Dữ liệu cấp phát** (`warmup.json`): BTC cung cấp một tập dữ liệu mẫu trích xuất từ tập huấn luyện chính thức. Bộ dữ liệu có cùng định dạng và cấu trúc với dữ liệu thật nhưng quy mô nhỏ hơn, hỗ trợ các đội viết data loader, xử lý đầu vào và chạy thử trước vòng chính thức.

**Submit thử nghiệm**: Các đội có thể nộp file dự đoán (`submission.zip`) lên hệ thống để kiểm tra tính hợp lệ và làm quen với quy trình.

**Tính chất điểm số**: Bài nộp ở vòng Warm-up không tính điểm chính thức, chỉ phục vụ mục đích kiểm thử và chuẩn bị cho các vòng thi tiếp theo.

