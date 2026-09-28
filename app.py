import streamlit as st
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. CẤU HÌNH TRANG GIAO DIỆN STREAMLIT
st.set_page_config(page_title="Chatbot Bài Giảng", page_icon="🙏", layout="centered")
st.title("🙏 HỎI - ĐÁP GIÁO LÝ")
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
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    return chain

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

    # Tìm kiếm các đoạn văn bản trong thư mục data/ giống với câu hỏi nhất
    docs = vector_store.similarity_search(user_question, k=4) # Lấy 4 đoạn liên quan nhất
    
    # Đưa các đoạn văn bản đó cho AI xử lý và sinh ra câu trả lời
    chain = get_conversational_chain()
    
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm ý trong bài giảng..."):
            context = "\n\n".join(doc.page_content for doc in docs)
            answer = chain.invoke({"context": context, "question": user_question})
            st.markdown(answer)
    
    # Lưu câu trả lời vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": answer})