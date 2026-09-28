import streamlit as st
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import ModelRateLimitError, ModelNotFoundError

# 1. CẤU HÌNH TRANG GIAO DIỆN STREAMLIT
st.set_page_config(page_title="Chatbot Bài Giảng", page_icon="🙏", layout="centered")
st.title("🙏 Trợ lý hỏi đáp Bài giảng của Quý Thầy")
st.write("Hãy đặt câu hỏi, tôi sẽ trả lời dựa trên các bài giảng đã được tải lên.")

# Lấy API Key từ phần cài đặt bảo mật của Streamlit (Secrets)
# LƯU Ý: Không bao giờ dán trực tiếp API key vào code để tránh bị lộ.
api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

# 2. HÀM ĐỌC DỮ LIỆU VÀ TẠO BỘ NHỚ VECTOR (Dùng cache để không load lại nhiều lần)
@st.cache_resource(show_spinner=True)
def load_and_process_data():
    documents = []
    # Quét tất cả các file .txt trong thư mục data/
    text_files = glob.glob("data/*.txt")
    
    if not text_files:
        return None

    # Đọc nội dung từng file
    for file_path in text_files:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
        
    # Băm nhỏ văn bản (mỗi đoạn 1000 ký tự) để AI dễ đọc hơn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Biến văn bản thành Vector và lưu vào FAISS cục bộ
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Chạy hàm tải dữ liệu
vector_store = load_and_process_data()

if vector_store is None:
    st.warning("Chưa có dữ liệu bài giảng. Bạn hãy tải các file .txt vào thư mục 'data/' nhé.")
    st.stop()

# 3. HÀM TẠO CHUỖI TRẢ LỜI CÂU HỎI (PROMPT)
# Các model Gemini có gói miễn phí, xếp theo thứ tự ưu tiên
FREE_MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash", "gemini-3.8-flash"]

def get_conversational_chain():
    # Căn dặn AI cách trả lời sao cho chuẩn mực và bám sát bài giảng
    prompt_template = """
    Bạn là một trợ lý ảo hỗ trợ Phật tử, được tạo ra để trả lời câu hỏi dựa trên các bài giảng của Quý Thầy.
    Hãy trả lời bằng giọng điệu từ bi, hòa ái, tôn trọng và dễ hiểu.
    Chỉ sử dụng thông tin trong phần "Ngữ cảnh (Context)" được cung cấp dưới đây để trả lời. 
    Nếu câu hỏi nằm ngoài ngữ cảnh bài giảng, hãy nhẹ nhàng nói rằng: "Dạ, trong phạm vi bài giảng hiện tại, Thầy chưa đề cập chi tiết đến vấn đề này. Mong bạn hoan hỷ đặt câu hỏi khác có liên quan ạ."
    Tuyệt đối không tự bịa ra kiến thức ngoài.

    Ngữ cảnh (Context):\n {context}?\n
    Câu hỏi:\n {question}\n

    Câu trả lời:
    """
    # Danh sách model, ưu tiên model có hạn mức miễn phí cao nhất.
    # Nếu model đầu hết lượt (429) hoặc không còn tồn tại (404) thì tự chuyển sang model kế tiếp.
    models = [
        ChatGoogleGenerativeAI(model=name, temperature=0.3, max_retries=1)
        for name in FREE_MODELS
    ]
    model = models[0].with_fallbacks(
        models[1:], exceptions_to_handle=(ModelRateLimitError, ModelNotFoundError)
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    return chain

# Ghi nhớ câu trả lời: cùng một câu hỏi được hỏi lại sẽ không tốn thêm lượt gọi API
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def answer_question(question):
    docs = vector_store.similarity_search(question, k=4) # Lấy 4 đoạn liên quan nhất
    context = "\n\n".join(doc.page_content for doc in docs)
    return get_conversational_chain().invoke({"context": context, "question": question})

# 4. GIAO DIỆN CHATBOT (LƯU LỊCH SỬ CHAT)
# Khởi tạo lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lại các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ô nhập tin nhắn của người dùng
user_question = st.chat_input("Nhập câu hỏi của bạn (ví dụ: Thầy dạy thế nào về lòng từ bi?)")

if user_question:
    # In câu hỏi của người dùng ra màn hình
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Tìm các đoạn bài giảng liên quan và đưa cho AI sinh ra câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm ý trong bài giảng..."):
            try:
                answer = answer_question(user_question.strip())
            except ModelRateLimitError:
                # Lỗi 429: API key đã hết lượt gọi (quota) của Google Gemini
                st.error("Dạ, hiện hệ thống đang quá tải hoặc đã hết lượt hỏi trong hôm nay. Mong bạn hoan hỷ đợi ít phút rồi hỏi lại ạ.")
                st.stop()
            st.markdown(answer)
    
    # Lưu câu trả lời vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": answer})