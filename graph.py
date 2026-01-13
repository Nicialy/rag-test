from neo4j import GraphDatabase
from typing import List
from base import CharacterInfo

class GenderGraphHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def save_characters(self, book_name: str, chars: List[CharacterInfo]):
        """Сохраняет персонажей и привязывает их к конкретной книге."""
        if not chars:
            return
        
        query = """
        MERGE (b:Book {title: $book_name})
        WITH b
        UNWIND $chars AS char_data
        MERGE (p:Person {name: char_data.name})
        ON CREATE SET 
            p.gender = char_data.gender,
            p.translation = char_data.translation,
            p.created_at = timestamp()
        ON MATCH SET 
            p.translation = COALESCE(p.translation, char_data.translation)
        
        MERGE (p)-[:APPEARS_IN]->(b)
        """
        
        chars_list = [
            {"name": c.name, "gender": c.gender, "translation": c.translation_name} 
            for c in chars if c.name
        ]
        
        with self.driver.session() as session:
            session.run(query, book_name=book_name, chars=chars_list)

    def get_graph_context(self, book_name: str, text_chunk: str) -> str:
        """Ищет персонажей конкретной книги, упомянутых в тексте."""
        found = []
        query = """
        MATCH (p:Person)-[:APPEARS_IN]->(b:Book {title: $book_name})
        RETURN p.name AS name, p.gender AS gender, p.translation AS translation
        """
        
        with self.driver.session() as session:
            result = session.run(query, book_name=book_name)
            for record in result:
                name = record["name"]
                if name.lower() in text_chunk.lower():  # Поиск без учета регистра
                    gender = record["gender"]
                    trans = record["translation"] or "нет перевода"
                    found.append(f"{name} -> ({trans}, Пол - {gender})")
        
        if not found:
            return f"В базе данных по книге '{book_name}' персонажей не найдено."
        
        return f"Контекст книги '{book_name}': " + "; ".join(found)

    def _clear_book(self, book_name: str):
        """Удаляет только одну книгу и связи её персонажей."""
        query = """
        MATCH (b:Book {title: $book_name})
        OPTIONAL MATCH (p:Person)-[r:APPEARS_IN]->(b)
        DELETE r, b
        """
        with self.driver.session() as session:
            session.run(query, book_name=book_name)
        print(f"Данные по книге '{book_name}' удалены.")