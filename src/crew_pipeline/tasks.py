from crewai import Task

def get_tasks(query: str, agents: dict, search_results: str):
    return [
        Task(
            name="Filter Content",
            description=f"""
            You are the Filter agent. Your job is to process the search results and extract the most relevant passages.

            The query is: '{query}'

            Here are the search results:
            {search_results}

            Your task is to:
            1. Read through all the search results
            2. Pick 3-5 of the most relevant chunks that answer the query or provide useful supporting context
            3. Ignore duplicates, errors, or off-topic content
            4. Return only a clean list of the most relevant passages

            Return ONLY a list of strings containing the most relevant passages, like this:
            [
                "passage 1...",
                "passage 2...",
                ...
            ]

            Important:
            - Do not include any metadata, scores, or source information
            - Do not include error messages or apologies
            - Do not make up or hallucinate content
            - If no relevant content is found, return an empty list []
            """,
            expected_output="A list of the most relevant passages as plain strings",
            agent=agents["filter"]
        ),
        Task(
            name="Generate Answer",
            description=f"""
            You are the Generator agent. Your task is to write a grounded, factual answer based on the filtered content.

            The query is: '{query}'

            Use only the information in the filtered passages.
            Do not hallucinate or invent facts.
            Provide a clear and concise answer in natural language.
            Answer only the user's questions, do not provide any more information than necessary to answer the question.


            If no content was filtered or the content doesn't contain enough information:
            1. Acknowledge that you don't have enough information
            2. Explain what information is available (if any)
            3. Suggest what additional information would be needed
            """,
            expected_output="A clear, factual answer based on the filtered content, or an acknowledgment of insufficient information",
            agent=agents["generator"]
        )
    ]
