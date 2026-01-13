import os
import asyncio
from pydantic_ai import Agent
from pydantic import Field, BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import model
from tenacity import retry, stop_after_attempt, wait_fixed
import re



# Используем ваш сплиттер для нарезки главы
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

class TranslationResponse(BaseModel):
    translated_text: str = Field(description="Перевод фрагмента главы")

translator_agent = Agent(
    model,
    output_type=TranslationResponse,
    system_prompt=(
        "Ты — экспертный переводчик на русский язык. Твоя задача — выполнить точный и естественный "
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

        # 2. Формируем запрос
        prompt = (
            f"РАБОТАЕМ НАД ГЛАВОЙ {chapter_name}\n"
            f"ТЕКУЩИЙ ФРАГМЕНТ ДЛЯ ПЕРЕВОДА:\n{chunk}"
        )
        final_translation = await translate(prompt,chunk)
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
    BOOKS_DIR = './corpus/books' 
    from test_epub import process_epub_files_bs4_v2
        # Загрузка данных
    data = process_epub_files_bs4_v2(BOOKS_DIR)
    book_find= False
    for book_name, chapters in data.items():
        i = 0
        if not book_find and book_name != "snedronningen_ru_en":
            continue
        else:
            book_find = True
        for chapter_name, pairs in chapters.items():
            if len(pairs['en']) == 0:
                continue
            print(f"Переводим главу: {chapter_name} в книге {book_name}...")
            try:
                res = await translate_chapter_text(pairs['en'],chapter_name,book_name)
                write_result2(res,0,i,book_name)
            except Exception as e:
                print(f"Критическая ошибка при переводе {chapter_name}: {e}")
            i+=1

if __name__ == "__main__":
    asyncio.run(main())
