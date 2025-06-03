from crewai import Agent

def get_generator_agent(llm):
    return Agent(
        role='Answer Generator',
        goal='Generate a grounded, factual answer using the provided context.',
        backstory=(
            "You are a helpful assistant that reads retrieved context chunks and answers the user’s query. "
            "Only use the provided context. Do not hallucinate.\n"
            "Output should be clear and concise. Include citations if available.\n"
        ),
        verbose=True,
        llm=llm
    )
