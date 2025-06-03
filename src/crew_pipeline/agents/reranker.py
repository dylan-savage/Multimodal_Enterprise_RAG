from crewai import Agent

def get_reranker_agent(llm):
    return Agent(
        role='Reranker',
        goal='Sort the retrieved documents based on relevance to the user query.',
        backstory=(
            "You receive a list of documents from different retrieval methods. "
            "Your job is to re-rank them based on how well they answer the user's query.\n"
            "Return the top N as a JSON list with updated scores:\n"
            "[{\"score\": ..., \"content\": \"...\"}, ...]\n"
            "Only return JSON, no explanation or markdown."
        ),
        verbose=True,
        llm=llm
    )