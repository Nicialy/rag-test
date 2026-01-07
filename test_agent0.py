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
from reader import EpubBilingualParser
from tenacity import retry, stop_after_attempt, wait_fixed
import re


CHROMA_PATH = "chroma_db_pydantic_ai_1"
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
EMBEDDINGS =  HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'} # или 'cpu', если нет видеокарты
)



@dataclass
class TranslationDeps:
    """Зависимости для агента: база данных и текущая глава."""
    vector_store: Chroma
    chapter_id: str

translator_agent = Agent(
    model,
    output_type=str,
    system_prompt=(
        "Ты — экспертный переводчик. Твоя задача — выполнить точный и естественный "
        "перевод предоставленного текста. Возвращай только перевод текста, без лишних слов."
    ),
)

def check_translate(original_text: str, translated_text: str, threshold_percent: float = 1.0) -> bool:
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
async def translate_chapter_text(chapter_text) -> str:
    prompt = f"""
        Переведи следующий текст: {chapter_text}
    """
    async with translator_agent.run_stream(prompt) as response:
        res_text = await response.get_output()
        if check_translate(chapter_text, res_text):
            raise Exception
    return res_text


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


async def main():
    # 1. Данные
    parser = EpubBilingualParser("./data")
    data = parser.get_all_books_data()
    i = 0
    for book_name, chapters in data.items():
        # started = False 
        if i ==0:
            i+=1
            continue
        for chapter_name, pairs in chapters.items():
            # 1. Логика пропуска глав (например, до 10-й)
            # if not started:
            #     # Ищем число 10 в названии главы (регуляркой или просто поиском)
            #     if "CHAPTER IX" in chapter_name:
            #         started = True
            #         print(f"--- Начинаем перевод с: {chapter_name} ---")
            #     else:
            #         # Пока не нашли 10-ю, просто идем дальше
            #         continue

            # 2. Основной процесс перевода (только когда started == True)
            full_chapter_en = "\n".join([p['en'] for p in pairs])
            
            if not full_chapter_en.strip():
                continue
                
            print(f"Переводим главу: {chapter_name} в книге {book_name}...")
            
            try:
                # Вызов твоего метода с декоратором retry
                res = await translate_chapter_text(full_chapter_en)
                save_chapter(book_name, chapter_name, res, agent_index=0)
            except Exception as e:
                print(f"Критическая ошибка при переводе {chapter_name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())