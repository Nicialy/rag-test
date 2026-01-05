import os
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
from pydantic_ai import Agent, RunContext
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import model


CHROMA_PATH = "chroma_db_pydantic_ai_1"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)



@dataclass
class TranslationDeps:
    """Зависимости для агента: база данных и текущая глава."""
    vector_store: Chroma
    chapter_id: str

translator_agent = Agent(
    model,
    deps_type=TranslationDeps,
    system_prompt=(
        "Ты — экспертный переводчик. Твоя задача — выполнить точный и естественный "
        "перевод предоставленного текста, используя ТОЛЬКО предоставленный "
        "контекст, который относится к текущей главе."
    ),
)

@translator_agent.system_prompt
async def add_context_to_prompt(ctx: RunContext[TranslationDeps]) -> str:
    """
    Динамически извлекает контекст из ChromaDB перед запуском генерации.
    Это заменяет RAG-пайплайн LangChain.
    """
    query = ctx.prompt
    
    docs = ctx.deps.vector_store.similarity_search(
        query, 
        k=4, 
        filter={"source": ctx.deps.chapter_id}
    )
    
    context_text = "\n".join([d.page_content for d in docs])
    return f"Контекст из текущей главы:\n{context_text}"

# --- ФУНКЦИИ ИНДЕКСАЦИИ ---
def index_source_chapters(chapter_texts: Dict[str, str]):
    print("🔄 Индексация исходного текста...")
    all_docs = []
    for chapter_id, text in chapter_texts.items():
        docs = TEXT_SPLITTER.create_documents([text], metadatas=[{"source": chapter_id}])
        all_docs.extend(docs)
    
    return Chroma.from_documents(
        documents=all_docs,
        embedding=EMBEDDINGS,
        persist_directory=CHROMA_PATH
    )

# --- ОСНОВНАЯ ЛОГИКА ---
async def translate_chapter(vector_store: Chroma, chapter_id: str, text: str):
    print(f"\n[Глава {chapter_id}] ➡️ Начало перевода...")
    
    deps = TranslationDeps(vector_store=vector_store, chapter_id=chapter_id)
    
    try:
        # Запуск агента
        result = await translator_agent.run(
            f"Переведи следующую часть текста: {text}",
            deps=deps
        )
        return result.data
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return f"ОШИБКА: {e}"

async def main():
    # 1. Данные
    SOURCE_CHAPTERS = {
        "Chapter_1": "Lady Catherine was a proud woman, known for her sharp wit. Mr. Darcy was a bachelor. He claimed he hated city life and often visited her estate.",
        "Chapter_2": "The next morning, Mrs. Reynolds, the kind housekeeper, showed Elizabeth Bennet around the grand house. Mr. Darcy waited for her on the porch, feeling a strange joy. He then told Mrs. Reynolds to prepare tea."
    }

    # 2. Индексация
    vector_store = index_source_chapters(SOURCE_CHAPTERS)

    # 3. Перевод главы 1
    q1 = "Lady Catherine was a proud woman. He claimed he hated city life."
    res1 = await translate_chapter(vector_store, "Chapter_1", q1)
    print(f"\nРезультат [Chapter_1]:\n{res1}")

    # 4. Перевод главы 2
    q2 = "Mrs. Reynolds showed Elizabeth Bennet around. He waited for her on the porch."
    res2 = await translate_chapter(vector_store, "Chapter_2", q2)
    print(f"\nРезультат [Chapter_2]:\n{res2}")

if __name__ == "__main__":
    asyncio.run(main())