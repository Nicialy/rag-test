import os
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import List, Dict

class EpubBilingualParser:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def get_all_books_data(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """Сканирует папку и возвращает структуру по главам (Синхронно)"""
        all_books = {}
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".epub"):
                path = os.path.join(self.folder_path, filename)
                try:
                    book = epub.read_epub(path)
                    all_books[filename] = self._extract_content(book)
                except Exception as e:
                    print(f"Ошибка в файле {filename}: {e}")
        return all_books

    def _is_new_chapter(self, text: str) -> bool:
        """Ищет заголовки типа 'Chapter I.' или 'Глава 1'"""
        return bool(re.search(r'^(Chapter|Глава)\s+[IVXLCDM\d]+', text, re.IGNORECASE))

    def _extract_content(self, book: epub.EpubBook) -> Dict[str, List[Dict[str, str]]]:
        chapters = {}
        current_chapter_name = "Intro" 
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # В синхронной версии используем .get_content() напрямую
                soup = BeautifulSoup(item.get_content(), "html.parser")
                paragraphs = soup.find_all('p')
                
                en_buffer = []
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if not text:
                        continue
                    
                    # Если нашли заголовок главы (EN часть)
                    if self._is_new_chapter(text) and not p.find('i'):
                        current_chapter_name = text
                        if current_chapter_name not in chapters:
                            chapters[current_chapter_name] = []
                        en_buffer = [] 
                        continue

                    # Если нашли перевод (курсив)
                    if p.find('i'):
                        if en_buffer:
                            en_text = " ".join(en_buffer)
                            
                            if current_chapter_name not in chapters:
                                chapters[current_chapter_name] = []
                            
                            chapters[current_chapter_name].append({
                                "en": en_text,
                                "ru": text
                            })
                            en_buffer = []
                    else:
                        # Накапливаем обычный английский текст
                        en_buffer.append(text)
                        
        return chapters
    

    def parse_chapter_content_with_paragraphs(self, html_content):
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

    def process_epub_files_bs4_v2(self,books_directory):
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
                            parsed_content = self.parse_chapter_content_with_paragraphs(content_html)
                            
                            # Сохраняем списки абзацев
                            chapters_data[chapter_title] = {
                                'book': item.file_name,
                                'en': '\n\n'.join(parsed_content['english_paragraphs']),
                                'ru': '\n\n'.join(parsed_content['russian_paragraphs'])
                            }
                            print(f"  -> Извлечена глава: {chapter_title} (Англ: {len(parsed_content['english_paragraphs'])}, Рус: {len(parsed_content['russian_paragraphs'])})")

                    all_books_data[book_title] = chapters_data
                    
                except Exception as e:
                    print(f"Не удалось открыть или обработать EPUB-файл {filename}: {e}")

        return all_books_data

# Пример использования:
# parser = EpubBilingualParser("./data")
# data = parser.get_all_books_data()
# print(data)
# Catriona ok 31
# THE american ok 26 глав
# The white comapny ok 38