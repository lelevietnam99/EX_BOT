import streamlit as st
import os
import re
import glob
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import (
    ModelRateLimitError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelInvalidRequestError,
)
from google import genai
from rank_bm25 import BM25Okapi

# 1. CẤU HÌNH TRANG GIAO DIỆN STREAMLIT
st.set_page_config(page_title="Chatbot Bài Giảng", page_icon="🙏", layout="centered")
st.title("🙏 Trợ lý hỏi đáp Bài giảng của Quý Thầy")
st.write("Hãy đặt câu hỏi, tôi sẽ trả lời dựa trên các bài giảng đã được tải lên.")

# Lấy API Key từ phần cài đặt bảo mật của Streamlit (Secrets)
# LƯU Ý: Không bao giờ dán trực tiếp API key vào code để tránh bị lộ.
api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

# 2. ĐỌC BÀI GIẢNG VÀ TẠO BỘ TÌM KIẾM
# Tìm đoạn bài giảng liên quan bằng từ khóa (thuật toán BM25), chạy ngay trên máy chủ:
# không cần gọi API, không tốn lượt miễn phí, khởi động gần như tức thì.
def tokenize(text):
    """Tách câu thành các từ, thêm cả cặp 2 từ liền nhau vì tiếng Việt có nhiều từ ghép (từ bi, thiền định...)."""
    words = re.findall(r"\w+", text.lower())
    return words + [a + "_" + b for a, b in zip(words, words[1:])]

class LectureSearch:
    def __init__(self, docs):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in docs])

    def search(self, question, k=5):
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in best]

@st.cache_resource(show_spinner="Đang đọc bài giảng...")
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
    return LectureSearch(docs)

# Chạy hàm tải dữ liệu
lecture_search = load_and_process_data()

if lecture_search is None:
    st.warning("Chưa có dữ liệu bài giảng. Bạn hãy tải các file .txt vào thư mục 'data/' nhé.")
    st.stop()

# 3. DANH SÁCH MODEL MIỄN PHÍ VÀ HÀM TRẢ LỜI CÂU HỎI
# Các model Gemini có gói miễn phí, xếp theo thứ tự ưu tiên (hạn mức free cao nhất lên trước).
# Model nào không còn tồn tại hoặc key không được dùng thì app tự bỏ qua.
FREE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]
AUTO = "🔄 Tự động (dùng lần lượt tất cả model free)"

# Lỗi của một model thì chuyển sang model kế tiếp:
# 429 hết lượt, 404 không tồn tại, 403 cần trả phí, 400 model không hỗ trợ
SKIP_ERRORS = (ModelRateLimitError, ModelNotFoundError, ModelPermissionDeniedError, ModelInvalidRequestError)

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_available_models():
    """Hỏi Google xem API key này dùng được những model nào, giữ lại các model free (Flash, Flash-Lite, Gemma)."""
    try:
        client = genai.Client(api_key=api_key)
        found = []
        for m in client.models.list():
            name = m.name.removeprefix("models/")
            if "generateContent" not in (m.supported_actions or []):
                continue
            if not ("flash" in name or name.startswith("gemma")):
                continue
            if any(x in name for x in ("tts", "image", "live", "audio", "embedding", "robotics", "computer-use")):
                continue
            found.append(name)
    except Exception:
        # Không lấy được danh sách thì dùng danh sách có sẵn
        return FREE_MODELS
    known = [name for name in FREE_MODELS if name in found]
    # Model mới Google vừa thêm (chưa có trong danh sách trên) được xếp sau: Flash-Lite, Gemma, rồi Flash
    extra = sorted(
        (name for name in found if name not in FREE_MODELS),
        key=lambda n: (0 if "lite" in n else 1 if n.startswith("gemma") else 2, n),
    )
    return known + extra or FREE_MODELS

# Ghi nhớ model vừa hết lượt để tạm bỏ qua 60 giây, đỡ mất thời gian chờ
@st.cache_resource
def exhausted_models():
    return {}

def get_conversational_chain(model_name):
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
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, max_retries=1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    return chain

# Ghi nhớ câu trả lời: cùng một câu hỏi được hỏi lại sẽ không tốn thêm lượt gọi API
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def answer_question(question, model_order):
    docs = lecture_search.search(question, k=5) # Lấy 5 đoạn liên quan nhất
    context = "\n\n".join(doc.page_content for doc in docs)

    skipped = exhausted_models()
    last_error = None
    for model_name in model_order:
        if skipped.get(model_name, 0) > time.time():
            continue
        try:
            answer = get_conversational_chain(model_name).invoke({"context": context, "question": question})
            return answer, model_name
        except SKIP_ERRORS as e:
            last_error = e
            if isinstance(e, ModelRateLimitError):
                skipped[model_name] = time.time() + 60
    raise last_error or ModelRateLimitError("Tất cả model đều đang hết lượt")

# 4. GIAO DIỆN CHATBOT (LƯU LỊCH SỬ CHAT)
# Ô chọn model ở thanh bên trái
available_models = get_available_models()
with st.sidebar:
    st.header("⚙️ Cài đặt")
    choice = st.selectbox("Chọn model AI", [AUTO] + available_models)
    st.caption(
        "Chế độ Tự động sẽ dùng lần lượt tất cả model miễn phí: model này hết lượt thì chuyển sang model khác. "
        "Nếu bạn chọn một model cụ thể, model đó được dùng trước, hết lượt mới chuyển sang model khác."
    )

if choice == AUTO:
    model_order = tuple(available_models)
else:
    model_order = (choice,) + tuple(m for m in available_models if m != choice)

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
                answer, used_model = answer_question(user_question.strip(), model_order)
            except ModelRateLimitError:
                # Lỗi 429: tất cả model đều đã hết lượt gọi (quota) của Google Gemini
                st.error("Dạ, hiện hệ thống đang quá tải hoặc đã hết lượt hỏi trong hôm nay. Mong bạn hoan hỷ đợi ít phút rồi hỏi lại ạ.")
                st.stop()
            except SKIP_ERRORS as e:
                st.error(f"Dạ, không gọi được model AI nào. Chi tiết lỗi: {e}")
                st.stop()
            st.markdown(answer)
            st.caption(f"Trả lời bởi: {used_model}")
    
    # Lưu câu trả lời vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": answer})