import os
import asyncio
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

# --- LangChain RAG & VectorStore ---
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.merger import MergerRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LangChain Pydantic & Extraction ---
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser

# --- КОНФИГУРАЦИЯ ---
load_dotenv()
if not os.getenv("MISTRAL_API_KEY"):
    print("❌ Ошибка: Переменная окружения MISTRAL_API_KEY не найдена. Проверьте файл .env")
    exit()

CHROMA_SOURCE = "chroma_db_baseline_source" # Хранилище исходного текста (для контекста сцены)
LLM_MODEL_NAME = "mistral-medium-2508" 
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Общие объекты
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# --- ГЛОБАЛЬНАЯ ГРАФОВАЯ БАЗА (Словарь для имитации) ---
# Ключ: Имя персонажа, Значение: Пол ("М", "Ж", "Н/Д")
GRAPH_STORE: Dict[str, str] = {}


class CharacterGender(BaseModel):
    """Модель для извлечения имени персонажа и его пола."""
    character_name: str = Field(description="Полное имя персонажа.")
    gender: str = Field(description="Пол персонажа: 'М' (мужской) или 'Ж' (женский).")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v.upper() not in ['М', 'Ж']:
            raise ValueError('Пол должен быть "М" или "Ж"')
        return v.upper()

class CharacterList(BaseModel):
    """Список всех извлеченных персонажей."""
    characters: List[CharacterGender]

# Создаем парсер на основе Pydantic-модели
ENTITY_PARSER = PydanticOutputParser(pydantic_object=CharacterList)


async def extract_and_populate_graph(chapter_texts: Dict[str, str]):
    """Извлекает персонажей из всего исходного текста и заполняет GRAPH_STORE."""
    global GRAPH_STORE
    
    # LLM для извлечения сущностей (включаем режим JSON)
    extractor_llm = ChatMistralAI(
        model=LLM_MODEL_NAME, 
        temperature=0, 
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Шаблон для извлечения
    EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         "Ты — эксперт по извлечению сущностей. Проанализируй текст и верни "
         "список всех персонажей и их пола ('М' или 'Ж') в формате JSON. "
         "Если пол неизвестен, проигнорируй персонажа.\n"
         "Формат вывода: {format_instructions}"),
        ("human", "Проанализируй следующий текст: {text}")
    ])
    
    # Цепочка извлечения
    extraction_chain = (
        EXTRACTION_PROMPT.partial(format_instructions=ENTITY_PARSER.get_format_instructions())
        | extractor_llm
        | ENTITY_PARSER
    )
    
    print("🌳 🔄 Запуск Entity Extractor для создания Графа...")
    
    # Проходим по всем главам для извлечения
    for chapter_id, text in chapter_texts.items():
        try:
            result: CharacterList = await extraction_chain.ainvoke({"text": text})
            
            # Заполняем глобальный Граф
            for char in result.characters:
                GRAPH_STORE[char.character_name] = char.gender
                
        except Exception as e:
            print(f"❌ Ошибка извлечения в главе {chapter_id}: {e}")

    print(f"✅ Создание Графа завершено. Всего персонажей: {len(GRAPH_STORE)}.")
    # print(f"DEBUG: Граф: {GRAPH_STORE}") # Для отладки

from langchain.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class SimpleGraphRetriever(BaseRetriever):
    """Простой ретривер, который ищет имена в запросе и возвращает их пол из GRAPH_STORE."""
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        
        # Шаг 1: Извлечение потенциальных имен из запроса
        # NOTE: Это упрощенный шаг; в реальной системе используется NER
        # Здесь мы просто проверяем, присутствует ли имя из GRAPH_STORE в запросе
        
        facts = []
        for name, gender in GRAPH_STORE.items():
            if name in query:
                # Шаг 2: Форматирование факта для LLM
                fact_text = f"ФАКТ О ПЕРСОНАЖЕ: Персонаж '{name}' имеет пол '{gender}'."
                facts.append(Document(page_content=fact_text, metadata={"source": "GraphStore"}))
                
        return facts

# Инициализируем Графовый ретривер
GRAPH_RETRIEVER = SimpleGraphRetriever()

def create_graph_rag_pipeline(current_chapter_id: str):
    """
    Создает RAG-пайплайн для GraphRAG, использующий Source Retriever и Graph Retriever.
    """
    llm = ChatMistralAI(model=LLM_MODEL_NAME, temperature=0.1)

    # 1. Ретривер ИСХОДНОГО ТЕКСТА (Source)
    source_store = Chroma(persist_directory=CHROMA_SOURCE, embedding_function=EMBEDDINGS)
    source_retriever = source_store.as_retriever(
        search_kwargs={"k": 2, "filter": {"source": current_chapter_id}}
    )
    
    # 2. Объединение ретриверов
    # Мы объединяем Source (семантика) и Graph (факты)
    lotr = MergerRetriever(retrievers=[source_retriever, GRAPH_RETRIEVER])
    
    # 3. Шаблон промпта (с акцентом на факты о поле)
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", 
         "Ты — экспертный переводчик. Используй предоставленный 'Объединенный Контекст'. "
         "Особенно обрати внимание на **ФАКТЫ О ПЕРСОНАЖАХ** для обеспечения "
         "абсолютной точности в роде и местоимениях при переводе.\n"
         "Объединенный Контекст: {context}"),
        ("human", "Переведи следующую часть текста: {question}")
    ])
    
    # 4. Создание цепочки с LCEL
    rag_chain = (
        {"context": lotr, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


# --- 4. ОСНОВНАЯ ЛОГИКА И ТЕСТЫ ---

# (Функция index_source_chapters должна быть перенесена из предыдущего кода,
#  так как она используется здесь для создания ChromaDB.)

async def run_chapter_translation_graph_rag(chapter_id: str, text_to_translate: str):
    """
    Выполняет перевод с использованием GraphRAG.
    """
    
    rag_chain = create_graph_rag_pipeline(chapter_id)
    print(f"\n[Глава {chapter_id}] ➡️ Начало перевода (GraphRAG)...")
    
    try:
        translated_text = await rag_chain.ainvoke(text_to_translate)
        print(f"[Глава {chapter_id}] ✅ Перевод завершен.")
        return translated_text
        
    except Exception as e:
        print(f"[Глава {chapter_id}] ❌ Ошибка LLM-запроса: {e}")
        return f"ОШИБКА: {e}"

# --- ТЕСТОВЫЙ ЗАПУСК ---

async def main():
    # 1. ИСХОДНЫЕ ДАННЫЕ
    SOURCE_CHAPTERS = {
        "Chapter_1": "Lady Catherine was a proud woman. Her cousin, Mr. Darcy, was a bachelor. He claimed he hated city life and often visited her estate.",
        "Chapter_2": "The next morning, Mrs. Reynolds, the kind housekeeper, showed Elizabeth Bennet around the grand house. Mr. Darcy waited for her on the porch, feeling a strange joy. He then told Mrs. Reynolds to prepare tea."
    }
    
    # 2. ИНДЕКСАЦИЯ ИСХОДНОГО ТЕКСТА (Vector Store)
    # (Предполагаем, что функция index_source_chapters из Агента 1 здесь доступна)
    # index_source_chapters(SOURCE_CHAPTERS) 
    
    # 3. СОЗДАНИЕ ГРАФА (НОВЫЙ ШАГ)
    await extract_and_populate_graph(SOURCE_CHAPTERS)
    
    # 4. --- Перевод с GraphRAG ---
    # QUERY_3 намеренно использует местоимения, чтобы проверить, работает ли Graph
    QUERY_3 = "Lady Catherine was a proud woman. Mr. Darcy claimed he hated city life, but he still visited. He was seeking solace."
    translation_3 = await run_chapter_translation_graph_rag(
        chapter_id="Chapter_1",
        text_to_translate=QUERY_3
    )
    print("\n" + "="*70)
    print(f"Перевод [Chapter_1] (GraphRAG): {translation_3}")
    print("="*70)

if __name__ == "__main__":
    # В реальном коде необходимо объединить функции Агента 1 и Агента 2
    # Для целей демонстрации здесь нужно вызвать: asyncio.run(main())
    pass