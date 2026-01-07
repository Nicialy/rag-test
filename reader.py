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

# Пример использования:
# parser = EpubBilingualParser("./data")
# data = parser.get_all_books_data()
# print(data)
# Catriona ok 31
# THE american ok 26 глав
# The white comapny ok 38