from crewai import Crew
from agents.clarifier import get_clarifier_agent
from agents.multi_retriever import get_multi_retriever_agent
from agents.reranker import get_reranker_agent
from agents.generator import get_generator_agent
from config.llm_config import get_llm
from crew_pipeline.tasks import get_tasks

llm = get_llm()

agents = {
    "clarifier": get_clarifier_agent(llm),
    "retriever": get_multi_retriever_agent(llm),
    "reranker": get_reranker_agent(llm),
    "generator": get_generator_agent(llm)
}

if __name__ == "__main__":
    query = input("Enter your query: ")
    tasks = get_tasks(query, agents)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=True
    )

    result = crew.kickoff()
    result_text = str(result)
    print("\nFinal Answer:\n", result_text)
