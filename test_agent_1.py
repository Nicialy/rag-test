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

# --- 1. Ваши настройки ---
CHROMA_PATH = "chroma_db_pydantic_ai_2"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'


hf_embeddings =  HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'} # или 'cpu', если нет видеокарты
)


# Используем ваш сплиттер для нарезки главы
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

class HFAdapter(chromadb.EmbeddingFunction): # Добавляем наследование
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # Важно: возвращаем именно список векторов
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
        "Ты — экспертный переводчик. Твоя задача — выполнить точный и естественный "
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

        # 1. Получаем контекст из прошлых глав вручную
        context = fetch_context_for_chunk(chunk,book_name)

        # 2. Формируем запрос
        prompt = (
            f"РАБОТАЕМ НАД ГЛАВОЙ {chapter_name}\n"
            f"{context}\n"
            f"ТЕКУЩИЙ ФРАГМЕНТ ДЛЯ ПЕРЕВОДА:\n{chunk}"
        )
        final_translation = await translate(prompt,chunk)
        save_to_memory(chunk, final_translation, chapter_name,book_name)
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


from base import LanguageCheckError, write_result2

async def main():
    # 1. Данные
    BOOKS_DIR = './corpus/books' 
    from test_epub import process_epub_files_bs4_v2
        # Загрузка данных
    data = process_epub_files_bs4_v2(BOOKS_DIR)
    for book_name, chapters in data.items():
        i = 0
        for chapter_name, pairs in chapters.items():
            if len(pairs['en']) == 0:
                continue
                
            print(f"Переводим главу: {chapter_name} в книге {book_name}...")
            
            try:
                res = await translate_chapter_text(pairs['en'],chapter_name,book_name)
                write_result2(res,1,i,book_name)
                #save_chapter(book_name, chapter_name, res, agent_index=1)
                i+=1
            except Exception as e:
                print(f"Критическая ошибка при переводе {chapter_name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
