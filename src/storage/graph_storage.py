from typing import Dict, List, Optional
from neo4j import GraphDatabase
from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class GraphStorage:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def store_extracted_data(self, data: Dict, source_file: str) -> None:

        with self.driver.session() as session:
            for entity in data.get("entities", []):
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.source_file = $source_file
                    """,
                    name=entity["name"],
                    type=entity.get("type", "UNKNOWN"),
                    source_file=source_file
                )

            for rel in data.get("relationships", []):
                session.run(
                    """
                    MATCH (subject:Entity {name: $subject})
                    MATCH (object:Entity {name: $object})
                    MERGE (subject)-[r:RELATES_TO {predicate: $predicate}]->(object)
                    SET r.source_file = $source_file
                    """,
                    subject=rel["subject"],
                    predicate=rel["predicate"],
                    object=rel["object"],
                    source_file=source_file
                )

    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.type = $type
                RETURN e.name as name, e.type as type, e.source_file as source_file
                """,
                type=entity_type
            )
            return [dict(record) for record in result]

    def get_relationships(self, entity_name: Optional[str] = None, direction: str = "outgoing") -> List[Dict]:
        with self.driver.session() as session:
            if entity_name:
                if direction == "outgoing":
                    query = """
                    MATCH (subject:Entity {name: $name})-[r:RELATES_TO]->(object:Entity)
                    RETURN subject.name as subject,
                        r.predicate as predicate,
                        object.name as object,
                        r.source_file as source_file
                    """
                else:  
                    query = """
                    MATCH (subject:Entity)-[r:RELATES_TO]->(object:Entity {name: $name})
                    RETURN subject.name as subject,
                        r.predicate as predicate,
                        object.name as object,
                        r.source_file as source_file
                    """
                result = session.run(query, name=entity_name)
            else:
                result = session.run("""
                    MATCH (subject:Entity)-[r:RELATES_TO]->(object:Entity)
                    RETURN subject.name as subject,
                        r.predicate as predicate,
                        object.name as object,
                        r.source_file as source_file
                """)
            return [dict(record) for record in result]


    def clear_database(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Cleared Neo4j database.")
