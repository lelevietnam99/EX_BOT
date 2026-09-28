import streamlit as st
import os
import re
import glob
import time
import hashlib
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.embeddings import Embeddings
from google import genai
from openai import OpenAI

# 1. CẤU HÌNH TRANG GIAO DIỆN STREAMLIT
st.set_page_config(page_title="Chatbot Bài Giảng", page_icon="🙏", layout="centered")
st.title("🙏 Trợ lý hỏi đáp Bài giảng của Quý Thầy")
st.write("Hãy đặt câu hỏi, tôi sẽ trả lời dựa trên các bài giảng đã được tải lên.")

# Lấy API Key từ phần cài đặt bảo mật của Streamlit (Secrets)
# LƯU Ý: Không bao giờ dán trực tiếp API key vào code để tránh bị lộ.
# GOOGLE_API_KEY là bắt buộc (dùng để đọc bài giảng). Các key khác không bắt buộc, có key nào thì dùng thêm nhà cung cấp đó.
api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

# 2. HÀM ĐỌC DỮ LIỆU VÀ TẠO BỘ NHỚ VECTOR (Dùng cache để không load lại nhiều lần)
EMBEDDING_MODEL = "models/gemini-embedding-001"
INDEX_DIR = "faiss_index"

class SlowEmbeddings(Embeddings):
    """Gói miễn phí của Google chỉ cho khoảng 100 đoạn văn/phút.
    Lớp này gửi từng nhóm nhỏ, gặp lỗi hết lượt (429) thì chờ đúng thời gian Google yêu cầu rồi gửi tiếp."""

    def __init__(self, batch_size=50, max_wait_rounds=20):
        self.base = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.batch_size = batch_size
        self.max_wait_rounds = max_wait_rounds

    def _with_retry(self, func, *args):
        for _ in range(self.max_wait_rounds):
            try:
                return func(*args)
            except Exception as e:
                message = str(e)
                if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
                    raise
                # Google cho biết cần chờ bao lâu (ví dụ "retry in 32.4s"), không có thì chờ 60 giây
                wait = re.search(r"retry in ([\d.]+)s", message)
                time.sleep(float(wait.group(1)) + 1 if wait else 60)
        return func(*args)

    def embed_documents(self, texts):
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors += self._with_retry(self.base.embed_documents, batch)
        return vectors

    def embed_query(self, text):
        return self._with_retry(self.base.embed_query, text)

def data_fingerprint(text_files):
    """Mã nhận dạng nội dung các file bài giảng: file đổi thì mã đổi, khi đó mới đọc lại."""
    h = hashlib.sha256(EMBEDDING_MODEL.encode())
    for file_path in sorted(text_files):
        h.update(file_path.encode())
        with open(file_path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]

@st.cache_resource(show_spinner="Đang đọc bài giảng lần đầu, có thể mất vài phút...")
def load_and_process_data():
    # Quét tất cả các file .txt trong thư mục data/
    text_files = glob.glob("data/*.txt")
    
    if not text_files:
        return None

    embeddings = SlowEmbeddings()

    # Nếu đã có bộ nhớ vector lưu sẵn cho đúng các file này thì dùng lại, không tốn lượt gọi Google
    index_path = os.path.join(INDEX_DIR, data_fingerprint(text_files))
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Đọc nội dung từng file
    documents = []
    for file_path in text_files:
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
        
    # Băm nhỏ văn bản (mỗi đoạn 1000 ký tự) để AI dễ đọc hơn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Biến văn bản thành Vector và lưu vào FAISS, đồng thời lưu ra ổ đĩa để lần sau dùng lại
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(index_path)
    return vector_store

# Chạy hàm tải dữ liệu
vector_store = load_and_process_data()

if vector_store is None:
    st.warning("Chưa có dữ liệu bài giảng. Bạn hãy tải các file .txt vào thư mục 'data/' nhé.")
    st.stop()

# 3. CÁC NHÀ CUNG CẤP AI MIỄN PHÍ
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

def is_free_gemini(name):
    """Giữ lại các model free của Google (Flash, Flash-Lite, Gemma), bỏ model giọng nói, hình ảnh..."""
    if not ("flash" in name or name.startswith("gemma")):
        return False
    return not any(x in name for x in ("tts", "image", "live", "audio", "embedding", "robotics", "computer-use"))

# Các nhà cung cấp khác đều dùng chuẩn API của OpenAI, chỉ khác địa chỉ (base_url).
# "secret": tên key trong Streamlit Secrets; "default": dùng khi không lấy được danh sách model;
# "keep": điều kiện để giữ một model (bỏ model giọng nói, kiểm duyệt, embedding...).
OPENAI_COMPATIBLE = {
    "Groq": {
        "secret": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "keep": lambda m: not any(x in m for x in ("whisper", "tts", "guard", "playai", "orpheus", "compound")),
    },
    "Cerebras": {
        "secret": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "default": ["llama-3.3-70b"],
        "keep": lambda m: True,
    },
    "Mistral": {
        "secret": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "default": ["mistral-small-latest", "mistral-large-latest"],
        "keep": lambda m: m.endswith("-latest") and not any(
            x in m for x in ("embed", "moderation", "ocr", "voxtral", "transcribe", "codestral", "devstral")
        ),
    },
    # OpenRouter gom nhiều model free (DeepSeek, Llama, Qwen, Gemma...), model free có đuôi ":free"
    "OpenRouter": {
        "secret": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default": ["deepseek/deepseek-r1:free", "meta-llama/llama-3.3-70b-instruct:free"],
        "keep": lambda m: m.endswith(":free"),
    },
}

def get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_gemini_models():
    """Hỏi Google xem API key này dùng được những model nào, giữ lại các model free."""
    try:
        client = genai.Client(api_key=api_key)
        found = [
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "generateContent" in (m.supported_actions or []) and is_free_gemini(m.name.removeprefix("models/"))
        ]
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

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_provider_models(provider):
    """Lấy danh sách model của một nhà cung cấp (Groq, OpenRouter...)."""
    config = OPENAI_COMPATIBLE[provider]
    try:
        client = OpenAI(api_key=get_secret(config["secret"]), base_url=config["base_url"])
        found = sorted(m.id for m in client.models.list() if config["keep"](m.id))
    except Exception:
        return config["default"]
    return found or config["default"]

def get_all_models():
    """Danh sách tất cả model dùng được, dạng (nhà cung cấp, tên model)."""
    models = [("Gemini", name) for name in get_gemini_models()]
    for provider, config in OPENAI_COMPATIBLE.items():
        if get_secret(config["secret"]):
            models += [(provider, name) for name in get_provider_models(provider)]
    return models

def label(model):
    return f"{model[0]} · {model[1]}"

# Ghi nhớ model vừa lỗi để tạm bỏ qua, đỡ mất thời gian chờ
@st.cache_resource
def failed_models():
    return {}

def get_conversational_chain(model):
    # Căn dặn AI cách trả lời sao cho chuẩn mực và bám sát bài giảng
    prompt_template = """
    Bạn là một trợ lý ảo hỗ trợ Phật tử, được tạo ra để trả lời câu hỏi dựa trên các bài giảng của Quý Thầy.
    Hãy trả lời bằng tiếng Việt, với giọng điệu từ bi, hòa ái, tôn trọng và dễ hiểu.
    Chỉ sử dụng thông tin trong phần "Ngữ cảnh (Context)" được cung cấp dưới đây để trả lời.
    Nếu câu hỏi nằm ngoài ngữ cảnh bài giảng, hãy nhẹ nhàng nói rằng: "Dạ, trong phạm vi bài giảng hiện tại, Thầy chưa đề cập chi tiết đến vấn đề này. Mong bạn hoan hỷ đặt câu hỏi khác có liên quan ạ."
    Tuyệt đối không tự bịa ra kiến thức ngoài.

    Ngữ cảnh (Context):\n {context}?\n
    Câu hỏi:\n {question}\n

    Câu trả lời:
    """
    provider, model_name = model
    if provider == "Gemini":
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, max_retries=1)
    else:
        config = OPENAI_COMPATIBLE[provider]
        llm = ChatOpenAI(
            model=model_name,
            api_key=get_secret(config["secret"]),
            base_url=config["base_url"],
            temperature=0.3,
            max_retries=1,
            timeout=60,
        )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm | StrOutputParser()
    return chain

# Ghi nhớ câu trả lời: cùng một câu hỏi được hỏi lại sẽ không tốn thêm lượt gọi API
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def answer_question(question, model_order):
    docs = vector_store.similarity_search(question, k=4) # Lấy 4 đoạn liên quan nhất
    context = "\n\n".join(doc.page_content for doc in docs)

    skipped = failed_models()
    last_error = None
    for model in model_order:
        if skipped.get(model, 0) > time.time():
            continue
        try:
            answer = get_conversational_chain(model).invoke({"context": context, "question": question})
        except Exception as e:
            # Model này lỗi thì chuyển sang model kế tiếp.
            # Hết lượt (429): bỏ qua 1 phút. Lỗi khác (sai key, model không còn...): bỏ qua 10 phút.
            last_error = e
            skipped[model] = time.time() + (60 if isinstance(e, ModelRateLimitError) else 600)
            continue
        # Các model "suy luận" như DeepSeek R1 có thể kèm phần <think>...</think>, bỏ phần đó đi
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        if answer:
            return answer, label(model)
    raise last_error or ModelRateLimitError("Tất cả model đều đang hết lượt")

# 4. GIAO DIỆN CHATBOT (LƯU LỊCH SỬ CHAT)
# Ô chọn model ở thanh bên trái
AUTO = "🔄 Tự động (dùng lần lượt tất cả model free)"
available_models = get_all_models()
labels = {label(m): m for m in available_models}
with st.sidebar:
    st.header("⚙️ Cài đặt")
    choice = st.selectbox("Chọn model AI", [AUTO] + list(labels))
    st.caption(
        "Chế độ Tự động sẽ dùng lần lượt tất cả model miễn phí: model này hết lượt thì chuyển sang model khác. "
        "Nếu bạn chọn một model cụ thể, model đó được dùng trước, hết lượt mới chuyển sang model khác."
    )
    connected = ["Gemini"] + [p for p, c in OPENAI_COMPATIBLE.items() if get_secret(c["secret"])]
    st.caption(f"Đang kết nối: {', '.join(connected)} ({len(available_models)} model)")

if choice == AUTO:
    model_order = tuple(available_models)
else:
    model_order = (labels[choice],) + tuple(m for m in available_models if m != labels[choice])

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
                # Lỗi 429: tất cả model đều đã hết lượt gọi
                st.error("Dạ, hiện hệ thống đang quá tải hoặc đã hết lượt hỏi trong hôm nay. Mong bạn hoan hỷ đợi ít phút rồi hỏi lại ạ.")
                st.stop()
            except Exception as e:
                st.error(f"Dạ, không gọi được model AI nào. Chi tiết lỗi: {e}")
                st.stop()
            st.markdown(answer)
            st.caption(f"Trả lời bởi: {used_model}")

    # Lưu câu trả lời vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": answer})