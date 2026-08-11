# TỔNG HỢP CHI TIẾT THÔNG TIN THI ĐẤU: NỘI DUNG 1 - LegalIR
**Cuộc thi Khoa học Dữ liệu UIT 2026 (UIT DSC 2026)**

---

## 1. MỤC TIÊU & BẢN CHẤT BÀI TOÁN
* **Mục tiêu:** Xây dựng phương pháp truy vấn văn bản pháp luật dựa trên mô hình ngôn ngữ lớn (LLMs) nhằm cải thiện khả năng tìm kiếm, hiểu và truy xuất thông tin pháp lý từ kho dữ liệu văn bản tiếng Việt.
* **Yêu cầu giải pháp:** Đảm bảo độ chính xác cao, khả năng mở rộng tốt và có tiềm năng ứng dụng thực tế trong các hệ thống hỗ trợ pháp luật thông minh tại Việt Nam.

---

## 2. Lưu ý trong quá trình tham gia cuộc thi liên quan đến công bố khoa học
* Các đội có thứ hạng trong cuộc thi sẽ được mời viết bài nghiên cứu công bố phương pháp của đội
* Do đây sẽ là một bài nghiên cứu sẽ được công bố ở tạp chí, nên BTC cần các nhóm lưu ý những điều sau:
  * **Nên có*** một giả thiết/giả định cho việc giải quyết tác vụ của cuộc thi. Sau đó sử dụng thực nghiệm để kiểm nghiệm giả thiết/giả định đó.
  * Quá trình thực nghiệm **nên có** đầy đủ các kịch bản để kiểm nghiệm giả thiết/giả định mà các đội đang theo đuổi.
  * **Nên có** những số liệu/phân tích cho kết quả của các phương pháp đã thử, vì sao phương pháp này chưa tốt? chưa tốt ở điểm nào? phương pháp sau tốt hơn là vì nó khắc phục được điểm yếu gì của phương pháp trước?

Câu hỏi mà BTC đặt ra cho các tác vụ của cuộc thi năm nay đang chờ câu trả lời: với nguồn tài nguyên tính toán hạn chế (chỉ vận hành được hệ thống có dưới 4 tỷ tham số), và lượng dữ liệu khiêm tốn (khoảng 10k điểm dữ liệu cho mỗi tác vụ), các phương pháp deep learning thuần túy có hiệu quả ra sao so với các phương pháp tận dụng sức mạnh của các mô hình ngôn ngữ lớn?

* **Lưu ý**: hướng tiếp cận không giới hạn ở mô hình/hệ thống mà còn có thể mở rộng ra cho các chiến lược xử lý dữ liệu, trong khuôn khổ quy định mà BTC đã ghi (không data augmentation, không dữ liệu ngoài).

---

## 3. MỐC THỜI GIAN QUAN TRỌNG (DỰ KIẾN 2026)
* **01/07 – 16/08/2026:** Đăng ký tham gia cuộc thi.
* **01/08 – 05/08/2026:** Vòng khởi động (*Warm-up phase*).
* **06/08 – 18/09/2026:** Vòng *Public Test*.
* **19/09 – 23/09/2026:** Vòng *Private Test*.
* **24/09 – 24/10/2026:** Các nhóm Top 10 viết bài báo khoa học.
* **25/10 – 30/10/2026:** Phản biện bài báo khoa học của các nhóm dự thi.
* **13/11/2026:** Bế mạc cuộc thi.

---

## 4. QUY ĐỊNH VỀ DỮ LIỆU VÀ MÔ HÌNH (QUAN TRỌNG)
* **Bộ dữ liệu:**
  * Chỉ khai thác bộ dữ liệu chính thức do BTC phát hành, dữ liệu công khai có giấy phép mở, hoặc dữ liệu do bên thứ ba cấp phép bằng văn bản.
  * **NGHIÊM CẤM:** Gán nhãn thủ công, thu thập dữ liệu ngoài trái phép, áp dụng các phương pháp tăng cường dữ liệu (*data augmentation*) từ nguồn bên ngoài, hoặc sử dụng dữ liệu chứa thông tin cá nhân nhạy cảm chưa ẩn danh (tuân thủ Luật Bảo vệ Dữ liệu Cá nhân).
  * Lưu ý: Dữ liệu dùng để pre-train các mô hình LLM mở không bị tính là vi phạm quy định dữ liệu nguồn ngoài.
* **Mô hình được phép:**
  * Tổng số lượng tham số của toàn bộ hệ thống sử dụng trong bài dự thi phải dưới 4 tỷ tham số. Giới hạn này được tính trên tất cả các thành phần của hệ thống, bao gồm (nhưng không giới hạn) mô hình sinh, mô hình embedding, reranker hoặc các mô hình khác nếu được sử dụng trong pipeline.
  * **Được phép**: Các mô hình được tạo bằng kỹ thuật distillation vẫn được phép sử dụng nếu bản thân mô hình/hệ thống sau khi distillation có tổng số tham số dưới 4 tỷ.
  * **Lưu Ý**:  các kỹ thuật tối ưu không gian bộ nhớ (như LoRA, quantization, ...), để các mô hình lớn hơn 4 tỷ tham số vận hành tương đương mô hình ít hơn 4 tỷ tham số về mặt lưu trữ và vận hành. **Không cho phép**
  * Phải **đăng ký trước với BTC** để được xét duyệt trước khi sử dụng.
  * **CẤM** sử dụng các mô hình ngôn ngữ lớn thương mại hoặc mã nguồn đóng (VD: OpenAI GPT-4, Claude, Gemini API, ...). **CẤM** mọi hình thức gọi API (kể cả API phi lợi nhuận) trong quá trình xây dựng giải pháp
  * Các kỹ thuật tối ưu bộ nhớ như LoRA, Quantization, GPTQ, AWQ, GGUF hoặc các kỹ thuật tương tự không làm thay đổi số lượng tham số của mô hình. Vì vậy, các mô hình/hệ thống có trên 4 tỷ tham số, dù đã được quantize hoặc tối ưu để giảm dung lượng lưu trữ hay tài nguyên vận hành, vẫn không được phép sử dụng.
  * *Bổ sung mô hình mới:* Đội thi phải gửi đề xuất trước 10 ngày so với hạn chót của vòng Private Test; BTC phản hồi trong 5 ngày làm việc.

---

## 5. QUY ĐỊNH NỘP BÀI & NGUYÊN TẮC CÔNG KHAI MÃ NGUỒN
* **Nộp bài Vòng Private Test:** Tối đa **03 lần/ngày**. Kết quả cuối cùng tính theo phương pháp cho điểm cao nhất trên tập Private Test.
* **Nền tảng CodaLab:** Nhóm trưởng đăng ký tài khoản CodaLab bằng email đã khai báo, đặt *Team Name* trùng tên đội 
* **Kiểm định & Mã nguồn (Top 10 Nội dung 1):**
  * Top 10 đội phải gửi **Docker image + Mã nguồn (giấy phép MIT)** để BTC tái lập kết quả trên Private Test. Đội không cung cấp sẽ bị hủy kết quả, thứ hạng và giải thưởng.
  * Khi có yêu cầu, phải cung cấp file log huấn luyện và cấu hình môi trường trong vòng **48 giờ**.
  * Hồ sơ nộp bài phải đính kèm **"Data Statement"** và **"Model Card"** (trình bày nguồn gốc dữ liệu, quy trình tiền xử lý, phương pháp huấn luyện, chỉ số đánh giá và rủi ro).

---

## 6. YÊU CẦU BÀI BÁO KHOA HỌC (TOP 10 & TOP 1-2-3)
* **Trình bày nghiên cứu:** Top 10 của subtask được yêu cầu viết toàn văn bài báo khoa học trình bày giải pháp trong khuôn khổ kỷ yếu Cuộc thi Khoa học Dữ liệu UIT 2026 (mỗi bài có ít nhất 02 phản biện).
* **Bắt buộc:** Các đội nằm trong **Top 1, Top 2, Top 3** bắt buộc phải viết và nộp bài báo khoa học theo quy định để được công nhận kết quả và trao giải chính thức.
* **Cơ hội:** Các bài báo được chấp thuận có cơ hội đăng tại kỷ yếu cuộc thi và được mời viết bài cho số đặc biệt của *Tạp chí Phát triển Khoa học và Công nghệ thuộc ĐHQG-HCM*.

---

## 7. Đóng gói & Tái lập thực nghiệm:
* Linh hoạt hình thức: Không bắt buộc chỉ dùng Docker; các đội có thể nộp qua GitHub (commit/push) hoặc file ZIP chứa mã nguồn/trọng số.  
* Hướng dẫn tái lập: Bắt buộc có file README chi tiết từng bước.  
* Tải trọng số khi chạy: Được phép truy cập Internet để tải trọng số mô hình mã nguồn mở trong quá trình chạy thực nghiệm.

---

## 8. CÁC QUY ĐỊNH PHÁP LÝ & KỶ LUẬT KHÁC
* **Sở hữu trí tuệ:** Bản quyền giải pháp thuộc về đội thi. Đội đạt giải cam kết công bố mã nguồn mở trong vòng 30 ngày kể từ ngày chung kết. Đồng thời cấp cho UIT quyền sử dụng phi thương mại để trưng bày/truyền thông.
* **Xử lý vi phạm:** Nghiêm cấm gian lận, đạo văn, tấn công hệ thống, trao đổi dữ liệu/mã nguồn không công khai. Vi phạm lần 1 bị trừ tối đa 20% tổng điểm hoặc cảnh cáo; tái phạm bị loại ngay lập tức.
* **Kênh liên hệ hỗ trợ:** Email BTC: `dsc@uit.edu.vn` (Phản hồi trong 48h đối với các yêu cầu thông thường, gửi khiếu nại trong vòng 48h từ khi công bố kết quả).
