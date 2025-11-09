import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# --- LangChain Компоненты ---
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List
from langchain_core.documents import Document

# --- КОНФИГУРАЦИЯ ---
load_dotenv()
if not os.getenv("MISTRAL_API_KEY"):
    print("❌ Ошибка: Переменная окружения MISTRAL_API_KEY не найдена. Проверьте файл .env")
    exit()

# Константы (оставляем только CHROMA_SOURCE)
CHROMA_PATH = "chroma_db_baseline_source" # Единое хранилище
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 
LLM_MODEL_NAME = "mistral-medium-2508" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Инициализация общего разбивателя текста
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)
# Инициализация общих эмбеддингов
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


## 🚀 1. ФУНКЦИИ ИНДЕКСАЦИИ (ОЧИЩЕННЫЕ)

def index_source_chapters(chapter_texts: Dict[str, str]):
    """Индексирует все главы исходного текста в ЕДИНОМ хранилище CHROMA_PATH."""
    print("🔄 Инициализация и индексация исходного текста для Baseline...")
    
    all_docs: List[Document] = []
    
    for chapter_id, text_content in chapter_texts.items():
        chapter_docs = TEXT_SPLITTER.create_documents(
            [text_content], 
            metadatas=[{"source": chapter_id}] 
        )
        all_docs.extend(chapter_docs)

    # Создаем единое хранилище
    Chroma.from_documents(
        documents=all_docs,
        embedding=EMBEDDINGS,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Индексация исходного текста завершена. Всего чанков: {len(all_docs)}.")


## 🧩 2. RAG-ПАЙПЛАЙН (ЧИСТЫЙ BASELINE)

def create_baseline_pipeline(current_chapter_id: str):
    """
    Создает RAG-пайплайн для Чистой Базовой Линии.
    Используется только один ретривер, ограниченный текущей главой.
    """
    llm = ChatMistralAI(model=LLM_MODEL_NAME, temperature=0.1)

    # 1. Инициализация ЕДИНОГО ретривера
    source_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
    
    # Фильтруем поиск: ищем контекст только в текущей главе
    retriever = source_store.as_retriever(
        search_kwargs={"k": 4, "filter": {"source": current_chapter_id}}
    )
    
    # 2. Шаблон промпта
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         "Ты — экспертный переводчик. Твоя задача — выполнить точный и естественный "
         "перевод предоставленного текста, используя ТОЛЬКО предоставленный "
         "контекст, который относится к текущей главе.\n"
         "Контекст: {context}"),
        ("human", "Переведи следующую часть текста: {question}")
    ])
    
    # 3. Создание цепочки с LCEL (один ретривер)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


## 🧪 3. ОСНОВНАЯ ЛОГИКА ЭКСПЕРИМЕНТА

async def run_chapter_translation(chapter_id: str, text_to_translate: str):
    """
    Выполняет перевод для Baseline. Никакой индексации перевода здесь нет!
    """
    
    # 1. Создание RAG-пайплайна
    rag_chain = create_baseline_pipeline(chapter_id)
    
    print(f"\n[Глава {chapter_id}] ➡️ Начало перевода (Чистый Baseline)...")
    
    try:
        translated_text = await rag_chain.invoke(text_to_translate)
        
        # Индексация перевода УДАЛЕНА
        
        print(f"[Глава {chapter_id}] ✅ Перевод завершен. (Без добавления в память).")
        return translated_text
        
    except Exception as e:
        print(f"[Глава {chapter_id}] ❌ Ошибка LLM-запроса: {e}")
        return f"ОШИБКА: {e}"

# --- ТЕСТОВЫЙ ЗАПУСК ---

async def main():
    # 1. ИСХОДНЫЕ ДАННЫЕ
    SOURCE_CHAPTERS = {
        "Chapter_1": "Lady Catherine was a proud woman, known for her sharp wit. Mr. Darcy was a bachelor. He claimed he hated city life and often visited her estate.",
        "Chapter_2": "The next morning, Mrs. Reynolds, the kind housekeeper, showed Elizabeth Bennet around the grand house. Mr. Darcy waited for her on the porch, feeling a strange joy. He then told Mrs. Reynolds to prepare tea."
    }
    
    # 2. ИНДЕКСАЦИЯ ИСХОДНОГО ТЕКСТА (ЕДИНОРАЗОВО)
    index_source_chapters(SOURCE_CHAPTERS)
    
    # 3. --- Перевод Главы 1 ---
    QUERY_1 = "Lady Catherine was a proud woman. He claimed he hated city life."
    translation_1 = await run_chapter_translation(
        chapter_id="Chapter_1",
        text_to_translate=QUERY_1
    )
    print("\n" + "="*70)
    print(f"Перевод [Chapter_1] (Baseline): {translation_1}")
    print("="*70)

    # 4. --- Перевод Главы 2 ---
    # Перевод Главы 2 не будет использовать перевод Главы 1.
    QUERY_2 = "Mrs. Reynolds showed Elizabeth Bennet around. He waited for her on the porch. He then told Mrs. Reynolds to prepare tea."

    translation_2 = await run_chapter_translation(
        chapter_id="Chapter_2",
        text_to_translate=QUERY_2
    )
    print("\n" + "="*70)
    print(f"Перевод [Chapter_2] (Baseline): {translation_2}")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())