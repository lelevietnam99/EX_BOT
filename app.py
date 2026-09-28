import streamlit as st
import os
import re
import glob
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import ModelRateLimitError
from google import genai
from openai import OpenAI
from rank_bm25 import BM25Okapi

# 1. CẤU HÌNH TRANG GIAO DIỆN STREAMLIT
st.set_page_config(page_title="Chatbot Bài Giảng", page_icon="🙏", layout="centered")
st.title("🙏 Trợ lý hỏi đáp Bài giảng của Quý Thầy")
st.write("Hãy đặt câu hỏi, tôi sẽ trả lời dựa trên các bài giảng đã được tải lên.")

# Lấy API Key từ phần cài đặt bảo mật của Streamlit (Secrets)
# LƯU Ý: Không bao giờ dán trực tiếp API key vào code để tránh bị lộ.
# GOOGLE_API_KEY là bắt buộc. Các key khác không bắt buộc, có key nào thì dùng thêm nhà cung cấp đó.
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
    # Quét tất cả các file .txt trong thư mục data/
    text_files = glob.glob("data/*.txt")
    
    if not text_files:
        return None

    # Đọc nội dung từng file
    documents = []
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
    docs = lecture_search.search(question, k=5) # Lấy 5 đoạn liên quan nhất
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