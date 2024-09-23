## Code RAG
-Thư mục model_rag gồm 2 hàm để khởi tạo 2 model text_embedding và LLM từ hugging_face

-Thư mục upload_docs gồm 2 hàm để load list pdf và để lưu các chunks vào vector_store

-Thư mục vector_store gồm 4 hàm:

1/Kết nối qdrant client và thông báo có bao nhiêu collections đã tồn tại 

2/Tạo collections mới nếu chưa có hoặc xóa dữ liệu collections đã có (nếu cần)

3/Tạo qdrant vector store 

4/ Tạo quy trình cho kết quả từ query

## DEMO
Chạy file demo_rag.py được:

-Kết quả chunking ở file DIAMOND _ 2K.json, Mentorship PLATINUM _10K.json

-Kết quả hỏi đáp rag ở file ANSWER_RAG.json

## Kết luận:
-Các thư viện pdf2text em tìm được tất cả đều bị lỗi output: vài từ có thể bị thêm dấu cách tạo thành các chữ cái vô nghĩa.

 ví dụ: "Sự h ỗ tr ợ từ các ch uyên gia hàng đ ầu cùng với khả năn g"

-Chunking theo RecursiveCharacterTextSplitter thì có thể kiểm soát được độ dài mỗi chunk, nhưng ngữ nghĩa trong mỗi chunk có thể không tốt (ví dụ chunk lấy câu cuối của đoạn 1 ghép với câu đầu của đoạn 2 trong khi đoạn 1 và 2 nói về ngữ nghĩa khác nhau).

-Chunking theo SemanticChunker thì khó kiểm soát được độ dài của mỗi chunk (langchain không có tham số hỗ trợ và nếu tự code để cắt thì cắt giữa chừng 1 chunk sẽ khiến ngữ nghĩa bị ảnh hưởng, rất có thể tạo ra 1 chunk rất dài 1 chunks cực kỳ ngắn).

-Chunking theo Agentic Chunking sử dụng LLMs thì model tốt thì cần gọi API, model open thì hên xui, code phức tạp, thời gian chunk lâu vì đợi LLM, khó kiểm soát được độ dài của mỗi chunk tương tự SemanticChunker.

-LLM để hỏi đáp thì ít có model open nào hiệu suất tốt cho cả tiếng anh lẫn tiếng việt.

-Chunking theo RecursiveCharacterTextSplitter thì có thể kiểm soát được độ dài của query, còn chunking theo 2 cách kia thì khó kiểm soát được.
## Câu hỏi
-Có cần giải quyết lỗi từ pdf2text không và giải quyết bằng cách nào. 

(Em có ý tưởng là tách hết văn bản thành các chữ và duyệt qua mỗi lần 3 chữ gộp lần lượt lại rồi xem nó có nghĩa không mà em nghĩ xử lý vậy bị lâu - hoặc cũng có thể bỏ vào LLM để nó tìm và tạo lại)

-Mình nên chunk theo loại nào.

-Text-embedding và LLM là anh tự chọn xong hết và sẽ chuyển nó qua onnx xong cho lên triton server phải không.

-Về vector store thì em xử lý như vậy có cần thêm bước nào nữa không, với em dùng from langchain.chains import RetrievalQA thì có ổn không ạ.