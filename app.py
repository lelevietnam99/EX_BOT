import glob
import os

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = st.secrets.get("CHAT_MODEL", "gemini-3.5-flash")  # có thể đổi trong Secrets
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

st.set_page_config(page_title="Chatbot Bài Giảng", page_icon="🙏", layout="centered")
st.title("🙏 HỎI - ĐÁP GIÁO LÝ")
st.write("Hãy đặt câu hỏi, tôi sẽ trả lời dựa trên các bài giảng đã được tải lên.")

# API key lấy từ Streamlit Secrets (không ghi trực tiếp vào code)
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Chưa cấu hình GOOGLE_API_KEY trong Settings → Secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cắt văn bản thành các đoạn ~size ký tự, ưu tiên ngắt ở xuống dòng/khoảng trắng."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_texts(texts, task_type):
    """Tạo embedding theo lô (tối đa 100 đoạn / lần gọi)."""
    vectors = []
    for i in range(0, len(texts), 100):
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i : i + 100],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in result.embeddings)
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)  # chuẩn hóa để dùng tích vô hướng


@st.cache_resource(show_spinner="Đang đọc bài giảng và tạo chỉ mục...")
def load_index():
    files = sorted(glob.glob("data/*.txt"))
    if not files:
        return None
    chunks = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            chunks.extend(split_text(f.read()))
    return chunks, embed_texts(chunks, "RETRIEVAL_DOCUMENT")


try:
    index = load_index()
except Exception as e:
    st.error(f"Không tạo được chỉ mục từ Gemini API: {e}")
    st.info(
        "Nếu lỗi là 403 SERVICE_DISABLED: hãy tạo key tại https://aistudio.google.com/apikey "
        "(hoặc bật 'Generative Language API' cho project của key), rồi cập nhật Secrets và Reboot app."
    )
    st.stop()

if index is None:
    st.warning("Chưa có dữ liệu bài giảng. Hãy đặt các file .txt vào thư mục 'data/'.")
    st.stop()

chunks, chunk_vectors = index

PROMPT = """Bạn là một trợ lý ảo hỗ trợ Phật tử, được tạo ra để trả lời câu hỏi dựa trên các bài giảng của Quý Thầy.
Hãy trả lời bằng giọng điệu từ bi, hòa ái, tôn trọng và dễ hiểu.
Chỉ sử dụng thông tin trong phần "Ngữ cảnh" dưới đây để trả lời.
Nếu câu hỏi nằm ngoài ngữ cảnh bài giảng, hãy nhẹ nhàng nói rằng: "Dạ, trong phạm vi bài giảng hiện tại, Thầy chưa đề cập chi tiết đến vấn đề này. Mong bạn hoan hỷ đặt câu hỏi khác có liên quan ạ."
Tuyệt đối không tự bịa ra kiến thức ngoài.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Câu trả lời:"""


def answer_question(question):
    q_vec = embed_texts([question], "RETRIEVAL_QUERY")[0]
    top = np.argsort(chunk_vectors @ q_vec)[::-1][:TOP_K]
    context = "\n\n".join(chunks[i] for i in top)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=PROMPT.format(context=context, question=question),
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text or "Dạ, hiện chưa có câu trả lời. Mong bạn thử lại ạ."


# Lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Nhập câu hỏi của bạn (ví dụ: Thầy dạy thế nào về lòng từ bi?)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm ý trong bài giảng..."):
            try:
                answer = answer_question(question)
            except Exception as e:
                answer = f"Xin lỗi, đã có lỗi khi gọi Gemini API: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})