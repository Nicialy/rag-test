from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document 
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter 
)

# --- Конфигурация (Остается прежней) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Или ваша модель
LLM_MODEL_NAME = "mistral-large-latest" # Или ваша модель Mistral
CHROMA_PATH = "./chroma_db"

# Инициализация эмбеддингов (глобально или внутри функции)
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_mezo_context_chunks(text: str) -> List[Document]:
    """Создает средние чанки для тональности и стиля (Mezo-Context)."""
    mezo_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separator="\n\n" 
    )
    return mezo_splitter.create_documents([text])

def create_micro_context_chunks(text: str) -> List[Document]:
    """Создает короткие чанки для локальной лексики и фраз (Micro-Context)."""
    micro_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    return micro_splitter.create_documents([text])

def create_macro_context_chunks(text: str) -> List[Document]:
    """Создает большие чанки для сюжетной консистентности (Macro-Context)."""
    macro_splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separator="\n\n"
    )
    return macro_splitter.create_documents([text])


def index_multi_level_chroma(full_text: str, unique_id: str) -> Dict[str, Chroma]:
    """
    Индексирует текст в три отдельные коллекции ChromaDB.
    unique_id используется, чтобы коллекции для каждой главы были уникальны.
    """
    vector_stores = {}
    # 1. Микро
    micro_chunks = create_micro_context_chunks(full_text)
    vector_stores['micro'] = Chroma.from_documents(
        documents=micro_chunks,
        embedding=EMBEDDINGS,
        collection_name=f"micro_{unique_id}" # Уникальное имя коллекции
    )
    # 2. Мезо
    mezo_chunks = create_mezo_context_chunks(full_text)
    vector_stores['mezo'] = Chroma.from_documents(
        documents=mezo_chunks,
        embedding=EMBEDDINGS,
        collection_name=f"mezo_{unique_id}"
    )
    # 3. Макро
    macro_chunks = create_macro_context_chunks(full_text)
    vector_stores['macro'] = Chroma.from_documents(
        documents=macro_chunks,
        embedding=EMBEDDINGS,
        collection_name=f"macro_{unique_id}"
    )
    return vector_stores

def get_multilevel_context_str(inputs: dict) -> str:
    """
    Функция-обертка для RunnableLambda.
    Принимает словарь с ключами 'chunk_to_translate' и 'vector_stores'.
    """
    chunk_to_translate = inputs["chunk_to_translate"]
    vector_stores = inputs["vector_stores"]
    
    retrieved_contexts = {}
    
    micro_docs = vector_stores['micro'].as_retriever(search_kwargs={"k": 2}).invoke(chunk_to_translate)
    retrieved_contexts['micro'] = " ".join([doc.page_content for doc in micro_docs])
    
    mezo_docs = vector_stores['mezo'].as_retriever(search_kwargs={"k": 1}).invoke(chunk_to_translate)
    retrieved_contexts['mezo'] = " ".join([doc.page_content for doc in mezo_docs])
    
    macro_docs = vector_stores['macro'].as_retriever(search_kwargs={"k": 1}).invoke(chunk_to_translate)
    retrieved_contexts['macro'] = " ".join([doc.page_content for doc in macro_docs])
    
    full_context_str = (
        f"--- МИКРО-КОНТЕКСТ (Лексика): {retrieved_contexts['micro']}\n"
        f"--- МЕЗО-КОНТЕКСТ (Стиль): {retrieved_contexts['mezo']}\n"
        f"--- МАКРО-КОНТЕКСТ (Сюжет): {retrieved_contexts['macro']}"
    )
    return full_context_str