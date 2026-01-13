import os
import asyncio
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional
from pydantic_ai import Agent, RunContext
from pydantic import Field, BaseModel
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import model
from tenacity import retry, stop_after_attempt, wait_fixed
import re
import chromadb

EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'


hf_embeddings =  HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'} # или 'cpu', если нет видеокарты
)
CHROMA_PATH = "chroma_db_pydantic_ai_3"
from graph import GenderGraphHandler
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "test12345"

from agents import extractor_agent
from base import write_result2, CharactersList


graph = GenderGraphHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

@retry(stop=stop_after_attempt(15), wait=wait_fixed(60))
async def extract_characters_to_db(text: str,book_name):
    """Этап 1: Анализ текста и наполнение графа."""
    try:
        result = await extractor_agent.run(f"Текст для анализа: {text}")
        characters_list: CharactersList = result.output
        graph.save_characters(book_name, characters_list.characters)
        print(">>> Анализ завершен. Данные в Neo4j обновлены.")
    except Exception as e:
        print(f"!!! Ошибка экстракции: {e}")
        raise e



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

class HFAdapter(chromadb.EmbeddingFunction): 
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return hf_embeddings.embed_documents(input)
    

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="translated",
    embedding_function=HFAdapter()
)


def fetch_context_for_chunk(chunk_text: str,book_name:str, n: int = 2) -> str:
    """Ищет в базе похожие куски текста из прошлых глав."""
    results = collection.query(query_texts=[chunk_text], n_results=n,where={"book": book_name})
    
    if not results['documents'] or not results['documents'][0]:
        return "В базе пока нет похожих фрагментов."

    context = "ПРИМЕРЫ СТИЛЯ ИЗ ПРОШЛЫХ ГЛАВ:\n"
    for original, meta in zip(results['documents'][0], results['metadatas'][0]):
        context += f"--- Оригинал ---\n{original}\n--- Твой перевод ---\n{meta['translation']}\n\n"
    return context

def save_to_memory(original_chunk: str, translated_chunk: str, chapter_num: int, book_name: str):
    collection.add(
        documents=[original_chunk],
        metadatas=[{
            "translation": translated_chunk,
            "chapter": chapter_num,
            "book": book_name,  
            "type": "chapter_chunk"
        }],
        ids=[str(uuid.uuid4())]
    )

class TranslationResponse(BaseModel):
    translated_text: str = Field(description="Перевод фрагмента главы")

translator_agent = Agent(
    model,
    output_type=TranslationResponse,
    system_prompt=(
        "Ты — экспертный переводчик на русский. Твоя задача — выполнить точный и естественный "
        "перевод предоставленного текста. Возвращай только перевод текста, без лишних слов."
    ),
)

def check_translate(original_text: str, translated_text: str, threshold_percent: float = 7.0) -> bool:
    """
    threshold_percent: разрешенный процент латиницы (по умолчанию 1%)
    """
    if not translated_text:
        return True

    # 1. Считаем количество латинских букв
    latin_chars = len(re.findall(r'[a-zA-Z]', translated_text))
    total_chars = len(translated_text)
    
    latin_ratio = (latin_chars / total_chars) * 100 if total_chars > 0 else 0

    # 2. Список фраз-исключений (которые точно можно оставить)
    exceptions = ["sumptibus moesti rei", "Chapter", "v."]
    temp_text = translated_text
    for ex in exceptions:
        temp_text = temp_text.replace(ex, "")

    # 3. Проверка на китайский (обычно это 100% ошибка, оставляем жесткой)
    if re.search(r'[\u4e00-\u9fff]', translated_text):
        print("Найдена китайщина!")
        return True

    # 4. Проверка на критическое превышение латиницы
    if latin_ratio > threshold_percent:
        # Ищем первое вхождение для лога, чтобы понять, что это
        match = re.search(r'[a-zA-Z]', temp_text)
        if match:
            pos = match.start()
            snippet = translated_text[max(0, pos-15) : pos+15]
            print(f"Слишком много латиницы ({latin_ratio:.2f}%). Найдено тут: ...{snippet}...")
        return True


    # 5. Проверка на полноту (соотношение длин)
    if len(translated_text) < (len(original_text) * 0.6):
        print("Текст подозрительно короткий")
        return True
    return False


@retry(stop=stop_after_attempt(30), wait=wait_fixed(60))
async def translate(prompt, chapter_text):
    async with translator_agent.run_stream(prompt) as response:
        res_text = await response.get_output()
        if check_translate(chapter_text, res_text.translated_text):
            raise Exception
        return res_text.translated_text


async def translate_chapter_text(chapter_text, chapter_name,book_name) -> str:
    chunks = text_splitter.split_text(chapter_text)
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Обработка фрагмента {i+1}/{len(chunks)}...")
        await extract_characters_to_db(chunk,book_name)
        # 1. Получаем контекст из прошлых глав вручную
        context_vector = fetch_context_for_chunk(chunk,book_name)
        context_graph = graph.get_graph_context(book_name, chunk)

        # 2. Формируем запрос
        prompt = (
            f"РАБОТАЕМ НАД ГЛАВОЙ {i}\n"
            f"Голосарий: \n {context_graph}\n"
            f"{context_vector}"
            f"ТЕКУЩИЙ ФРАГМЕНТ ДЛЯ ПЕРЕВОДА:\n{chunk}"
        )
        final_translation = await translate(prompt,chunk)
        save_to_memory(chunk, final_translation, chapter_name, book_name)
        translated_chunks.append(final_translation)

    full_translated_chapter = "\n\n".join(translated_chunks)
    return full_translated_chapter


def save_chapter(book_name: str, chapter_name: str, content: str, agent_index: int = 0):
    """
    Сохраняет переведенный текст главы в отдельный файл.
    Структура: output/Название_Книги/Номер_Агента/Глава.txt
    """
    # 1. Очистка имен от запрещенных символов
    clean_book_name = re.sub(r'[\\/*?:"<>|]', "", book_name).strip()
    clean_chapter_name = re.sub(r'[\\/*?:"<>|]', "", chapter_name).strip()

    output_dir = os.path.join("output", clean_book_name, str(agent_index))
    
    # Создаем всю цепочку директорий
    os.makedirs(output_dir, exist_ok=True)

    # 3. Путь к итоговому файлу
    file_path = os.path.join(output_dir, f"{clean_chapter_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Успешно сохранено (Агент {agent_index}): {file_path}")


from base import write_result2

async def main():
    # 1. Данные
    BOOKS_DIR = './corpus/books' 
    from test_epub import process_epub_files_bs4_v2
        # Загрузка данных
    data = process_epub_files_bs4_v2(BOOKS_DIR)
    book_find = False
    chapter_find=False
    for book_name, chapters in data.items():
        i = 0
        if not book_find and book_name != "pinocchio_en_ru":
            continue
        else:
            book_find = True
        for chapter_name, pairs in chapters.items():
            if len(pairs['en']) == 0:
                continue
            if i >= 32 or chapter_find:
                chapter_find = True
                print(f"Переводим главу: {chapter_name} в книге {book_name}...")
                
                try:
                    res = await translate_chapter_text(pairs['en'],chapter_name,book_name)
                    write_result2(res,3,i,book_name)
                    #save_chapter(book_name, chapter_name, res, agent_index=1)
                except Exception as e:
                    print(f"Критическая ошибка при переводе {chapter_name}: {e}")
            i+=1

if __name__ == "__main__":
    asyncio.run(main())
