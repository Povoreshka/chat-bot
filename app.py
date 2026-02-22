import streamlit as st
import os
import shutil
import tempfile
import time
import gc
import re
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def clean_text(text):
    if not text:
        return text
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\.{5,}', '...', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_answer(relevant_results):
    if not relevant_results:
        return "Информация не найдена. Попробуйте переформулировать запрос."
    
    answer = ""
    
    for i, (doc, score) in enumerate(relevant_results, 1):
        page = doc.metadata.get('page', '?')
        content = clean_text(doc.page_content)
        
        if len(content) > 300:
            content = content[:300] + "..."
        
        answer += f"""
        <div style="background: #ffffff; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border: 1px solid #eaeef2;">
            <div style="display: flex; gap: 1rem; margin-bottom: 0.5rem; font-size: 0.75rem; color: #6b7280;">
                <span>Стр. {page}</span>
                <span>•</span>
                <span>{score:.0%} совпадение</span>
            </div>
            <div style="color: #1f2937; font-size: 0.9rem; line-height: 1.5;">{content}</div>
        </div>"""
    
    return answer

def safe_remove_database(db_path):
    if not os.path.exists(db_path):
        return True
    
    if 'vectorstore' in st.session_state:
        st.session_state.vectorstore = None
    
    gc.collect()
    time.sleep(1)
    
    for attempt in range(3):
        try:
            shutil.rmtree(db_path)
            return True
        except PermissionError:
            if attempt < 2:
                time.sleep(2)
            else:
                try:
                    temp_name = f"{db_path}_old_{int(time.time())}"
                    os.rename(db_path, temp_name)
                    return True
                except:
                    return False
        except:
            return False
    return False

def create_unique_database_path(base_path="./chroma_db"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_path}_{timestamp}"

def clear_all_data():
    st.session_state.vectorstore = None
    st.session_state.documents = None
    st.session_state.chunks = None
    st.session_state.pdf_filename = None
    st.session_state.chat_history = []
    st.session_state.processing_complete = False
    st.session_state.error_message = None
    
    gc.collect()
    time.sleep(1)
    
    if st.session_state.db_path and os.path.exists(st.session_state.db_path):
        safe_remove_database(st.session_state.db_path)
    
    if os.path.exists("./chroma_db"):
        safe_remove_database("./chroma_db")
    
    st.session_state.db_path = None

st.set_page_config(
    page_title="Чат с конспектом",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: #f9fafb;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .header {
        margin-bottom: 2rem;
    }
    
    .header h1 {
        font-size: 1.8rem;
        font-weight: 500;
        color: #111827;
        margin: 0;
    }
    
    .header p {
        color: #6b7280;
        font-size: 0.9rem;
        margin: 0.25rem 0 0;
    }
    
    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    .sidebar-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 1.5rem 0;
        padding: 1rem;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    
    .stat-card {
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 500;
        color: #111827;
    }
    
    .stat-label {
        font-size: 0.7rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    
    .file-info {
        background: #f9fafb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .file-info .filename {
        font-weight: 500;
        color: #111827;
    }
    
    .file-info .filesize {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    .message {
        margin-bottom: 1.5rem;
        animation: fadeIn 0.2s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
    }
    
    .user-message-content {
        background: #eef2ff;
        color: #1f2937;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 4px 12px;
        max-width: 70%;
        font-size: 0.95rem;
    }
    
    .bot-message {
        display: flex;
        justify-content: flex-start;
    }
    
    .bot-message-content {
        background: #ffffff;
        color: #1f2937;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 12px 4px;
        max-width: 70%;
        border: 1px solid #e5e7eb;
        font-size: 0.95rem;
    }
    
    .message-time {
        font-size: 0.65rem;
        color: #9ca3af;
        margin-top: 0.25rem;
        text-align: right;
    }
    
    .welcome-message {
        text-align: center;
        padding: 3rem 2rem;
        background: #f9fafb;
        border-radius: 8px;
        border: 1px dashed #d1d5db;
        margin: 2rem 0;
    }
    
    .welcome-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .welcome-text {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    .notification {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .notification-success {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    
    .notification-error {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }
    
    .stButton button {
        background: #ffffff;
        color: #374151;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 400;
        transition: all 0.1s;
        width: 100%;
    }
    
    .stButton button:hover {
        background: #f9fafb;
        border-color: #9ca3af;
    }
    
    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
        background: #ffffff;
        color: black;
    }
    
    .stTextInput input:focus {
        border-color: #9ca3af;
        box-shadow: none;
    }
    
    .streamlit-expanderHeader {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #374151;
        padding: 0.5rem 0.75rem;
    }
    
    .messages-container {
        margin: 1.5rem 0;
    }
    
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e5e7eb;
    }
    
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.documents = None
    st.session_state.chunks = None
    st.session_state.pdf_filename = None
    st.session_state.chat_history = []
    st.session_state.processing_complete = False
    st.session_state.error_message = None
    st.session_state.db_path = None

with st.sidebar:
    st.markdown('<div class="sidebar-title">Настройки</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Загрузить PDF",
        type=['pdf'],
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    if uploaded_file and not st.session_state.processing_complete:
        file_size = uploaded_file.size / 1048576
        st.markdown(f"""
        <div class="file-info">
            <div class="filename">📄 {uploaded_file.name}</div>
            <div class="filesize">{file_size:.1f} MB</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Обработка..."):
            try:
                progress_bar = st.progress(0)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                progress_bar.progress(10)
                
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                if not documents:
                    raise Exception("PDF не содержит текста")
                
                progress_bar.progress(30)
                
                for i, doc in enumerate(documents):
                    doc.metadata['page'] = i + 1
                
                documents = [doc for doc in documents if doc.page_content and len(doc.page_content.strip()) > 50]
                
                if not documents:
                    raise Exception("Нет страниц с текстом")
                
                chunk_size = st.session_state.get('chunk_size', 250)
                chunk_overlap = st.session_state.get('chunk_overlap', 30)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                chunks = text_splitter.split_documents(documents)
                
                if not chunks:
                    raise Exception("Не удалось создать фрагменты")
                
                progress_bar.progress(50)
                
                max_chunks = st.session_state.get('max_chunks', 100)
                if len(chunks) > max_chunks:
                    chunks = chunks[:max_chunks]
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                progress_bar.progress(70)
                
                if st.session_state.db_path and os.path.exists(st.session_state.db_path):
                    safe_remove_database(st.session_state.db_path)
                
                db_path = create_unique_database_path()
                
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )
                vectorstore.persist()
                
                progress_bar.progress(100)
                
                st.session_state.vectorstore = vectorstore
                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.pdf_filename = uploaded_file.name
                st.session_state.processing_complete = True
                st.session_state.error_message = None
                st.session_state.db_path = db_path
                
                os.unlink(tmp_path)
                
                st.success("Готово")
                st.rerun()
                
            except Exception as e:
                st.session_state.error_message = str(e)
                st.error(f"Ошибка: {str(e)}")
    
    if st.session_state.processing_complete:
        with st.expander("Параметры"):
            st.session_state.k_results = st.slider("Результатов", 1, 5, 3)
            st.session_state.min_relevance = st.slider("Мин. совпадение", 0.0, 1.0, 0.2, 0.05)
            st.session_state.chunk_size = st.slider("Размер фрагмента", 100, 500, 250, 50)
            st.session_state.chunk_overlap = st.slider("Перекрытие", 0, 100, 30, 10)
    
    if st.button("🗑️ Очистить всё", use_container_width=True):
        clear_all_data()
        st.rerun()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>Чат с конспектом</h1>
    <p>Задайте вопрос по загруженному документу</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.error_message:
    st.markdown(f"""
    <div class="notification notification-error">
        {st.session_state.error_message}
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.processing_complete:
    st.markdown("""
    <div class="welcome-message">
        <div class="welcome-icon">📄</div>
        <div class="welcome-text">Загрузите PDF-файл в меню слева</div>
    </div>
    """, unsafe_allow_html=True)
else:
    total_chars = sum(len(c.page_content) for c in st.session_state.chunks)
    avg_len = total_chars // len(st.session_state.chunks) if st.session_state.chunks else 0
    
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.documents)}</div>
            <div class="stat-label">страниц</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.chunks)}</div>
            <div class="stat-label">фрагментов</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_chars//1000}K</div>
            <div class="stat-label">символов</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_len}</div>
            <div class="stat-label">ср. длина</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="messages-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; color: #9ca3af; padding: 2rem;">
            Введите вопрос, чтобы начать диалог
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div class="user-message-content">
                        {message["content"]}
                        <div class="message-time">{message["time"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <div class="bot-message-content">
                        {message["content"]}
                        <div class="message-time">{message["time"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        question = st.text_input(
            "",
            placeholder="Введите вопрос...",
            label_visibility="collapsed",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("→", use_container_width=True)
    
    if ask_button and question:
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "time": current_time
        })
        
        with st.spinner("Поиск..."):
            try:
                k_results = st.session_state.get('k_results', 3)
                min_relevance = st.session_state.get('min_relevance', 0.2)
                
                results = st.session_state.vectorstore.similarity_search_with_relevance_scores(
                    question, k=k_results
                )
                
                if results:
                    relevant_results = [(doc, score) for doc, score in results if score >= min_relevance]
                    
                    if relevant_results:
                        answer = format_answer(relevant_results)
                    else:
                        answer = "Результаты с низким совпадением. Попробуйте переформулировать вопрос."
                else:
                    answer = "Ничего не найдено."
                    
            except Exception as e:
                answer = f"Ошибка: {str(e)}"
        
        st.session_state.chat_history.append({
            "role": "bot",
            "content": answer,
            "time": datetime.now().strftime("%H:%M")
        })
        
        st.rerun()

st.markdown('<div class="footer">Чат-бот для конспектов</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)