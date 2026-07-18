# Luồng xử lý AI-native

![Luồng xử lý logic của giải pháp AI-native](./ai-processing-flow.svg)

```mermaid
flowchart TD
    U["Khách hàng gửi yêu cầu"] --> A["AI hiểu mục đích, nhu cầu<br/>và ngữ cảnh hội thoại"]
    A --> B{"Đủ thông tin để tư vấn?"}

    B -->|Chưa đủ| C["Xác định thông tin còn thiếu"]
    C --> D["Đặt câu hỏi làm rõ"]
    D --> E["Khách hàng bổ sung thông tin"]
    E --> F["Cập nhật hồ sơ nhu cầu"]
    F --> B

    B -->|Đã đủ| G{"Loại yêu cầu hiện tại?"}
    G -->|Nhu cầu mới hoặc thay đổi| H["Tìm sản phẩm phù hợp<br/>từ dữ liệu doanh nghiệp"]
    G -->|Hỏi chi tiết hoặc so sánh| I["Sử dụng lại sản phẩm<br/>trong ngữ cảnh hiện tại"]
    G -->|Muốn thêm lựa chọn| J{"Còn lựa chọn phù hợp<br/>chưa giới thiệu?"}

    J -->|Còn| K["Đánh giá lại các lựa chọn còn lại"]
    J -->|Không còn| H
    H --> L["AI đánh giá và xếp hạng"]
    I --> M["AI tạo câu trả lời có căn cứ"]
    K --> L
    L --> M

    M --> N["Đề xuất sản phẩm<br/>Lý do phù hợp • Điểm cần cân nhắc"]
    N --> O["Ghi nhớ nhu cầu và kết quả tư vấn"]
    O --> P["Tiếp tục hội thoại theo ngữ cảnh"]
    P --> A
```

AI chỉ tạo đề xuất dựa trên dữ liệu sản phẩm của doanh nghiệp. Khi thông tin chưa đủ, AI chủ động làm rõ trước khi tư vấn; khi khách hàng hỏi tiếp, AI tận dụng ngữ cảnh đã có thay vì bắt đầu lại.
