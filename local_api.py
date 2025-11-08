import os
from pathlib import Path
from dotenv import load_dotenv

# --- Установите: pip install "pydantic-ai-slim[mistral]" python-dotenv markitdown[pdf] chromadb langchain-text-splitters langchain-huggingface ---

# --- RAG/Индексация компоненты ---
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ИСПРАВЛЕННЫЙ ИМПОРТ для Sentence Transformers через LangChain:
from chromadb.utils import embedding_functions
from chromadb import PersistentClient

# --- Pydantic AI/LLM компоненты ---
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool
from typing import List, Optional

# --- КОНФИГУРАЦИЯ ---
load_dotenv()
DATA_PATH = Path("./my_data") 
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = 'multi-qa-mpnet-base-dot-v1'
LLM_MODEL_NAME = "mistral:mistral-small-latest" 
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

if not os.getenv("MISTRAL_API_KEY"):
    print("❌ Ошибка: Переменная окружения MISTRAL_API_KEY не найдена. Проверьте файл .env")
    exit()

# --- СТРУКТУРЫ ДАННЫХ ---

class RetrievedContext(BaseModel):
    """Структура для хранения извлеченных фрагментов контекста."""
    chunk: str
    source: str

class RAGResponse(BaseModel):
    """Структура для финального ответа, включая источники."""
    answer: str = Field(description="Подробный ответ на вопрос, основанный только на предоставленном контексте.")
    sources: List[str] = Field(description="Список уникальных названий исходных документов, из которых был получен контекст.")

# --- 1. ФУНКЦИИ ИНДЕКСАЦИИ И CHROMADB ---
def create_or_load_vector_db():
    """
    Создает или загружает ChromaDB Collection и заполняет ее документами, если она пуста.
    """
    db_client = PersistentClient(path=CHROMA_PATH)
    
    # 1. Инициализация функции эмбеддингов

    collection_name = "mistral_rag_data"
    embeddings_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device='cpu' # Указываем устройство
    )

    # ИСПОЛЬЗУЕМ get_or_create_collection, чтобы гарантировать, что коллекция существует
    collection = db_client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embeddings_function
    )
    
    # Проверяем, пуста ли коллекция
    if collection.count() > 0:
        print(f"✅ Векторная база данных '{collection_name}' загружена. Документов: {collection.count()}.")
        return collection
    
    # --- Если коллекция пуста, начинаем индексацию ---
    print(f"🔄 Коллекция '{collection_name}' пуста. Начинается индексация...")
    
    # --- Загрузка и разбиение ---
    if not DATA_PATH.exists() or not any(DATA_PATH.iterdir()):
         print(f"❌ Ошибка: Папка с данными {DATA_PATH} не найдена или пуста. Индексация невозможна.")
         # Если нет данных, но база пуста, возвращаем пустую коллекцию (RAG-запросы будут неэффективны)
         return collection 

    md = MarkItDown()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_texts = []
    all_metadatas = []
    
    for file_path in DATA_PATH.iterdir():
        if file_path.is_file():
            try:
                result = md.convert(file_path)
                doc_chunks = text_splitter.split_text(result.text_content)
                
                for i, chunk in enumerate(doc_chunks):
                    all_texts.append(chunk)
                    all_metadatas.append({"source": str(file_path.name), "chunk_id": f"{file_path.name}_{i}"})
                    
            except Exception as e:
                print(f"   - ⚠️ Не удалось обработать {file_path.name}: {e}")
    
    # --- Добавление данных в коллекцию ---
    if not all_texts:
        print("🚨 Индексация прервана: нет обработанных чанков.")
        return collection
        
    print(f"✂️ Создано {len(all_texts)} чанков. Добавление в Chroma...")
    
    collection.add(
        documents=all_texts,
        metadatas=all_metadatas,
        ids=[f"id_{i}" for i in range(len(all_texts))]
    )
    print(f"✅ Индексация завершена. {collection.count()} документов в коллекции.")
    return collection
# --- 2. RAG TOOL ДЛЯ PYDANTIC AI ---

rag_collection = create_or_load_vector_db()

class RetrievalTool(Tool):
    name: str = "document_retriever"
    description: str = "Используйте этот инструмент, чтобы найти и извлечь наиболее релевантную информацию из внутренней базы знаний (документов) перед ответом на вопрос пользователя."

    def __init__(self):
        super().__init__(
            function=self.run, # <-- Ключевое ИСПРАВЛЕНИЕ: передача ссылки на метод run
            name="document_retriever", 
            description="Используйте этот инструмент, чтобы найти и извлечь наиболее релевантную информацию из внутренней базы знаний (документов) перед ответом на вопрос пользователя."
        )


    def run(self, query: str) -> List[RetrievedContext]:
        results = rag_collection.query(
            query_texts=[query],
            n_results=4,
            include=['documents', 'metadatas']
        )
        
        context_list = []
        if results and results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context_list.append(RetrievedContext(
                    chunk=doc,
                    source=meta['source']
                ))
        
        return context_list

# --- 3. ФУНКЦИЯ ЗАПРОСА ---

async def query_rag_system(question: str) -> RAGResponse:
    print("=" * 70)
    print(f"➡️ Получен запрос: {question}")

    agent = Agent(LLM_MODEL_NAME, tools=[RetrievalTool()],  output_type=RAGResponse)

    try:
        async with agent.run_stream(question) as response:
            return await response.get_output()

    except Exception as e:
        print(f"❌ Ошибка при выполнении RAG-запроса: {e}")
        return RAGResponse(
            answer=f"Извините, произошла техническая ошибка при обработке запроса: {e}",
            sources=[]
        )

# --- ЗАПУСК ---

if __name__ == "__main__":
    import asyncio
    print("Проверка готовности к RAG-запросам...")
    async def main():
        response1 = await query_rag_system("Какие основные шаги необходимо предпринять для развертывания RAG-системы?")
        print(f"\n[Ответ Mistral]: {response1.answer}")
        print(f"[Источники]: {response1.sources}")

        response2 = await query_rag_system("ЧТо ты можешь рассказать о Паше?")
        print(f"\n[Ответ Mistral]: {response2.answer}")
        print(f"[Источники]: {response2.sources}")
    asyncio.run(main())