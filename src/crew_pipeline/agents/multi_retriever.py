from crewai import Agent
from crew_pipeline.tools import (
    vector_retrieval_tool,
    graph_retrieval_tool,
    keyword_retrieval_tool
)

def get_multi_retriever_agent(llm):
    return Agent(
        role='Retriever',
        goal='Retrieve relevant documents using graph traversal, keyword search, and vector similarity.',
        backstory=(
            "You are a hybrid search agent that combines three retrieval methods: "
            "1. Graph traversal from Neo4j for relational knowledge, "
            "2. Keyword search from Whoosh for exact match, "
            "3. Semantic vector retrieval from Qdrant for related meaning.\n"
            "Always retrieve from all three, tag results by method, and output as a JSON list:\n"
            "[{\"source\": \"graph\", \"score\": ..., \"content\": \"...\"}, ...]\n"
            "Only return JSON — no extra commentary."
        ),
        tools=[
            vector_retrieval_tool,
            keyword_retrieval_tool,
            graph_retrieval_tool
        ],
        verbose=True,
        llm=llm
    )
