import requests
import xml.etree.ElementTree as ET
import zipfile
import io
import re

# --- Настройки ---
# Ссылка на ZIP-архив TMX из коллекции OPUS
# ВНИМАНИЕ: OPUS-файлы часто очень большие. 
# В данном примере используется ссылка на небольшой корпус (The Man Who Would Be King) для теста.
# Вы можете заменить её на другую ссылку, например:
# 'https://object.pouta.csc.fi/OPUS-Books/v1/tmx/ru-en.zip'
TMX_ZIP_URL = 'https://object.pouta.csc.fi/OPUS-Books/v1/tmx/TheManWhoWouldBeKing-ru-en.tmx.zip' 
# Имя TMX-файла внутри ZIP-архива
TMX_FILENAME = 'TheManWhoWouldBeKing-ru-en.tmx'

def download_and_parse_opus_tmx(url, filename):
    """
    Скачивает ZIP-архив, извлекает TMX-файл и парсит его.
    """
    print(f"Загрузка архива: {url}...")
    try:
        # Скачиваем ZIP-файл
        response = requests.get(url)
        response.raise_for_status() # Проверка на ошибки HTTP

        # Открываем ZIP-архив в памяти
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Извлекаем содержимое TMX-файла (обычно это единственный файл)
            if filename not in zf.namelist():
                 raise FileNotFoundError(f"Файл {filename} не найден в архиве.")
                 
            tmx_content = zf.read(filename).decode('utf-8')
            print("Архив успешно загружен и TMX-файл извлечен.")

            # Парсим XML-содержимое TMX
            root = ET.fromstring(tmx_content)
            return root
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке: {e}")
        return None
    except FileNotFoundError as e:
         print(e)
         return None
    except ET.ParseError as e:
        print(f"Ошибка при парсинге TMX: {e}")
        return None

def split_tmx_into_chapters(tmx_root):
    """
    Разделяет параллельные тексты на "главы" на основе тегов <seg>, содержащих 
    слова 'CHAPTER', 'ГЛАВА', 'PREFACE', 'ПРОЛОГ' и т.п.
    """
    
    # Пространство имен TMX (обычно 'http://www.lisa.org/tmx14')
    # Ищем его в корневом элементе
    ns_match = re.match(r'\{.*\}', tmx_root.tag)
    ns = ns_match.group(0) if ns_match else ''
    
    # Ищем все элементы Translation Unit
    tus = tmx_root.findall(f'.//{ns}tu') 
    
    chapters = []
    current_chapter_ru = []
    current_chapter_en = []
    chapter_index = 0
    
    # Регулярное выражение для обнаружения заголовков глав (нечувствительно к регистру)
    # Ищем слова типа: CHAPTER, ГЛАВА, PROLOGUE, PREFACE и т.п.
    chapter_pattern = re.compile(r'\b(CHAPTER|ГЛАВА|PROLOGUE|ПРЕДИСЛОВИЕ|SECTION|РАЗДЕЛ|PART|ЧАСТЬ)\b', re.IGNORECASE)

    for tu in tus:
        ru_seg = None
        en_seg = None
        
        # Находим сегменты для каждого языка внутри <tu>
        for tuv in tu.findall(f'{ns}tuv'):
            lang = tuv.get(f'{ns}lang') or tuv.get('lang') # Языковой атрибут может быть без неймспейса
            seg_element = tuv.find(f'{ns}seg')
            
            if seg_element is not None:
                text = seg_element.text.strip() if seg_element.text else ""
                
                # Упрощённый способ определить язык по атрибуту
                if 'ru' in lang:
                    ru_seg = text
                elif 'en' in lang:
                    en_seg = text

        # Проверяем, является ли сегмент заголовком главы
        is_chapter_title = False
        if ru_seg and chapter_pattern.search(ru_seg):
             is_chapter_title = True
        elif en_seg and chapter_pattern.search(en_seg):
             is_chapter_title = True

        # Если нашли заголовок главы И это не первая "глава" в файле
        if is_chapter_title and (current_chapter_ru or current_chapter_en):
            # Сохраняем предыдущую главу
            chapters.append({
                'chapter_index': chapter_index,
                'russian_text': "\n".join(current_chapter_ru),
                'english_text': "\n".join(current_chapter_en)
            })
            # Начинаем новую главу
            current_chapter_ru = []
            current_chapter_en = []
            chapter_index += 1
        
        # Добавляем сегменты в текущую главу
        if ru_seg and en_seg:
             current_chapter_ru.append(ru_seg)
             current_chapter_en.append(en_seg)


    # Добавляем последнюю главу, если в ней есть контент
    if current_chapter_ru or current_chapter_en:
        chapters.append({
            'chapter_index': chapter_index,
            'russian_text': "\n".join(current_chapter_ru),
            'english_text': "\n".join(current_chapter_en)
        })
        
    return chapters

def main():
    """Главная функция для выполнения процесса."""
    # 1. Скачивание и парсинг TMX
    tmx_root = download_and_parse_opus_tmx(TMX_ZIP_URL, TMX_FILENAME)
    
    if tmx_root is not None:
        # 2. Разделение на главы
        chapters = split_tmx_into_chapters(tmx_root)
        
        print("\n" + "="*50)
        print(f"✅ Успешно найдено {len(chapters)} 'глав' по разметке TMX.")
        print("="*50)
        
        # 3. Вывод первых нескольких глав для демонстрации
        for i, chapter in enumerate(chapters[:3]):
            print(f"\n### Глава {i+1} (Сегменты {chapter['chapter_index']}) ###")
            
            ru_lines = chapter['russian_text'].split('\n')
            en_lines = chapter['english_text'].split('\n')
            
            # Выводим первые 5 строк главы для наглядности
            print("--- Русский (первые 5 строк) ---")
            print("\n".join(ru_lines[:5]))
            print("--- English (first 5 lines) ---")
            print("\n".join(en_lines[:5]))

# Запускаем скрипт
if __name__ == "__main__":
    main()