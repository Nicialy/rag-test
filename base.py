import re
from pydantic import BaseModel, Field
from typing import List, Optional
import os

def is_chapter_clean(text: str) -> bool:
    """
    Проверяет, содержит ли глава нежелательные символы или языки.
    Считаем, что чистый текст должен быть в основном на русском.
    
    Критерии нечистоты:
    1. Обнаружение символов, характерных для китайского языка (очень широкий диапазон).
    2. Обнаружение большого количества латинских символов (английский/другие).
    """
    if not text:
        return False
    
    text_len = len(text)
    
    # 1. Поиск китайских/азиатских символов (широкий диапазон CJK)
    cjk_pattern = re.compile(r'[\u3000-\u9fff\uac00-\ud7af]')
    if cjk_pattern.search(text):
        return False
        
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    if latin_count / text_len > 0.05:
        return False

    return True


class LanguageCheckError(Exception):
    pass


class CharacterInfo(BaseModel):
    name: str = Field(description="Имя персонажа как оно встречается в тексте (например, 'Mr. Darcy').")
    translation_name: str = Field(description="Перевод Имени на русский язык")
    gender: str = Field(description="Пол: 'MALE', 'FEMALE' или 'UNKNOWN'. Определять строго по контексту (местоимения he/she).")

class CharactersList(BaseModel):
    characters: List[CharacterInfo] = Field(description="Список всех персонажей из текста.")


class FusionQuestionOut(BaseModel):
    first_query: str = Field(description="1 Предложение дял поиска в векторном хранилище")
    second_query: str = Field(description="2 Предложение дял поиска в векторном хранилище")
    third_qury: str = Field(description="3 Предложение дял поиска в векторном хранилище")



def write_result2(text, agent_number, filename, book_name):
    folder = f"agents/{agent_number}/{book_name}"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)