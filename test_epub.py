import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Указываем путь к папке books
BOOKS_DIR = './corpus/books' 

def parse_chapter_content_with_paragraphs(html_content):
    """
    Парсит HTML-контент главы. Удаляет ВСЕ HTML-теги (включая <i>) 
    и использует символ переноса строки ('\n') для разделения строк внутри <p>.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    chapter_data = {'russian_paragraphs': [], 'english_paragraphs': []}
    
    for p_tag in soup.find_all('p'):
        lang_code = p_tag.get('lang')
        
        # *** Ключевое изменение: Используем .get_text() с разделителем. ***
        # separator='\n' заменяет все теги <br/> на символы переноса строки.
        clean_text = p_tag.get_text(separator='\n', strip=True) 
        
        if clean_text:
            if lang_code == 'ru':
                chapter_data['russian_paragraphs'].append(clean_text)
            elif lang_code == 'en':
                chapter_data['english_paragraphs'].append(clean_text)
    
    return chapter_data

def process_epub_files_bs4_v2(books_directory):
    """
    Сканирует директорию, открывает каждый EPUB-файл, 
    извлекает контент глав и парсит его, сохраняя абзацы.
    """
    all_books_data = {}
    print(f"Поиск EPUB-файлов в: {books_directory}")
    
    for filename in os.listdir(books_directory):
        if filename.endswith(".epub"):
            filepath = os.path.join(books_directory, filename)
            book_title = filename[:-5]
            print(f"\n--- Обработка книги: {book_title} ---")
            
            try:
                book = epub.read_epub(filepath)
                chapters_data = {}
                
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        content_bytes = item.get_content()
                        content_html = content_bytes.decode('utf-8', errors='ignore')
                        
                        soup = BeautifulSoup(content_html, 'html.parser')
                        header = soup.find(['h1', 'h2', 'h3'])
                        chapter_title = header.get_text(strip=True) if header else f"Неизвестная Глава ({item.file_name})"
                        
                        # Парсим, используя новую функцию, которая сохраняет абзацы
                        parsed_content = parse_chapter_content_with_paragraphs(content_html)
                        
                        # Сохраняем списки абзацев
                        chapters_data[chapter_title] = {
                            'source_file': item.file_name,
                            'en': '\n\n'.join(parsed_content['english_paragraphs']),
                            'ru': '\n\n'.join(parsed_content['russian_paragraphs'])
                        }
                        print(f"  -> Извлечена глава: {chapter_title} (Англ: {len(parsed_content['english_paragraphs'])}, Рус: {len(parsed_content['russian_paragraphs'])})")

                all_books_data[book_title] = chapters_data
                
            except Exception as e:
                print(f"Не удалось открыть или обработать EPUB-файл {filename}: {e}")

    return all_books_data


def read_file(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    chapters = []
    for ch in soup.find_all("chapter"):
        text = "\n\n".join(p.get_text(strip=True) for p in ch.find_all("p"))
        if text:
            chapters.append({"id": ch.get("id"), "name": ch.get("name"), "text": text})
    return chapters

# Запуск обработки
if __name__ == "__main__":
    
    if not os.path.exists(BOOKS_DIR):
        print(f"Ошибка: Директория '{BOOKS_DIR}' не найдена. Проверьте путь.")
    else:
        results = process_epub_files_bs4_v2(BOOKS_DIR)
        
        # Пример вывода структуры результата
        print("\n===== Сводка по результатам =====")
        for book_name, chapters in results.items():
            print(f"Книга: **{book_name}**")
            for chapter_title, data in chapters.items():
                # Выводим первые 50 символов для примера
                print(f"  Глава: {chapter_title}")
                print(f"    Английский (начало): {data['english'][:50]}...")
                print(f"    Русский (начало): {data['russian'][:50]}...")
        
        # Теперь 'results' содержит всю необходимую информацию