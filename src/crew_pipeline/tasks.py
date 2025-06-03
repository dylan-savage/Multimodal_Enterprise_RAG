from crewai import Task

def get_tasks(query, agents):
    return [
        Task(
            name="Clarify Query",
            description=f"""
            You are the Clarifier agent assigned to evaluate this user query: '{query}'

            If the query is clear, respond with:
            {{"needs_clarification": false}}

            If unclear, respond with:
            {{"needs_clarification": true, "question": "..." }}

            Respond with ONLY valid JSON. Do not add commentary, markdown, or explanation.
            """,
            expected_output="A JSON object with 'needs_clarification' (boolean) and 'question' (string).",
            agent=agents["clarifier"]
        ),
        Task(
            name="Retrieve Content",
            description=f"""
            You are the Retriever agent. Your job is to retrieve relevant context based on the user's query.

            The query is: '{query}'

            Use hybrid retrieval:
            - Graph traversal for structured knowledge
            - Keyword filtering for exact matches
            - Vector search for semantically related content

            Return a consolidated set of relevant chunks.
            """,
            expected_output="A list of relevant chunks from graph, keyword, and vector retrieval.",
            agent=agents["retriever"]
        ),
        Task(
            name="Rerank Results",
            description=f"""
            You are the Reranker agent. Your job is to rank the retrieved chunks in terms of their relevance to the query.

            The query is: '{query}'

            Rank the chunks from most to least relevant.
            Eliminate duplicates and prioritize factual, high-coverage information.
            """,
            expected_output="A ranked list of content chunks in JSON format.",
            agent=agents["reranker"]
        ),
        Task(
            name="Generate Answer",
            description=f"""
            You are the Generator agent. Your task is to write a grounded, factual answer based on the ranked content.

            The query is: '{query}'

            Use only the information in the retrieved context.
            Do not hallucinate or invent facts.
            Provide a clear and concise answer in natural language.
            """,
            expected_output="A natural language answer to the original query.",
            agent=agents["generator"]
        )
    ]
