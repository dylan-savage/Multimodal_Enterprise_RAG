from crewai import Agent
from typing import Dict, Any, List

def get_filter_agent(llm: Any) -> Agent:
    """
    Creates a filter agent that processes search results and extracts relevant passages.
    
    Args:
        llm: The language model to use for the agent
        
    Returns:
        Agent: A configured filter agent
    """
    return Agent(
        role='Content Filter',
        goal='Extract the most relevant passages from search results to answer the user query',
        backstory="""You are a filtering agent.

You receive:
1. A user query (a question).
2. A blob of noisy multi-source text output from multiple search tools, including score-based annotations.

Your task:
- Read the full text blob.
- Pick 3–5 of the most relevant chunks that answer the user's query or provide useful supporting context.
- Ignore duplicates, errors, or off-topic content.
- Return only a clean list of the most relevant passages, as plain strings (no JSON, no metadata).

Respond ONLY with a list like this:
[
"passage 1...",
"passage 2...",
...
]""",
        verbose=True,
        llm=llm,
        allow_delegation=False
    ) 