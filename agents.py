import asyncio
import os
from typing import List, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_fixed

# --- Pydantic AI ---
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.models.mistral import MistralModel, MistralModelSettings

# Ваши локальные импорты (предполагаем, что они есть)
from base import is_chapter_clean, CharacterInfo, CharactersList, FusionQuestionOut

# --- КОНФИГУРАЦИЯ ---
LLM_MODEL_NAME = "mistral-medium" # Проверьте точное название для API, обычно это mistral-medium или mistral-large-latest
MISTRAL_API_KEY = "wdzW8nYbhlfgllaaogtJ5pB32G4XbGJo" # Лучше брать из os.environ
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "test12345"
from config import model

# Агент поиска персонажей
extractor_agent = Agent(
    model,
    output_type=CharactersList,
    system_prompt=(
        "Ты — эксперт по лингвистическому анализу. Твоя задача — извлечь персонажей и их грамматический род.\n"
        "Правила определения рода:\n"
        "1. He/Him/His -> MALE\n"
        "2. She/Her -> FEMALE\n"
        "3. It/Its (если это активный персонаж, например 'The Monster', 'The AI') -> NEUTER\n"
        "4. Если пол неизвестен или существо бесполое -> NEUTER"
        "Извлекай только именнованные сущности"
    )
)

# --- Агент 2: Переводчик ---
translator_agent = Agent(
    model,
    output_type=str,
    system_prompt=(
        "Ты — профессиональный литературный переводчик на русский язык. \n"
        "Тебе будет предоставлен КОНТЕКСТ с полом персонажей. Используй его, чтобы сохранить верные окончания.\n"
        "--- ИНСТРУКЦИИ ПО ВЫВОДУ ---\n"
        "1. Твоя задача — вернуть ТОЛЬКО переведенный текст.\n"
        "2. ЗАПРЕЩЕНО писать вступления и примечания.\n"
        "3. ЗАПРЕЩЕНО повторять оригинальный текст.\n"
        "4. Если перевод не требуется, верни пустую строку."
    )
)

query_generator_agent = Agent(
            model,
            output_type=FusionQuestionOut,
            system_prompt=(
                "Ты AI-ассистент. Сгенерируй 3 РАЗНЫХ поисковых запроса на АНГЛИЙСКОМ языке, "
                "чтобы найти похожий контекст/стиль для предоставленного текста в базе данных.\n"
                "1. Keywords search.\n2. Thematic description.\n3. Abstract summary.\n"
            )
        )
