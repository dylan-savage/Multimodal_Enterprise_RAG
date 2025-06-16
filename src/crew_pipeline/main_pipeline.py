from crewai import Crew
from src.crew_pipeline.agents.filter import get_filter_agent
from src.crew_pipeline.agents.generator import get_generator_agent
from src.crew_pipeline.tasks import get_tasks
from src.retrieval.hybrid_retrieval import get_formatted_results
from src.storage.qdrant_client import QdrantStorage
from src.storage.graph_storage import GraphStorage
from src.storage.keyword_index import KeywordIndex
from src.query_processing.query_rewriter import QueryRewriter
from config.llm_config import get_llm
from query_processing.query_classifier import QueryClassifier
from evaluation.evaluator import evaluate_response
import json


# Configuration
ENABLE_EVALUATION = False  # Set to True to enable evaluation

def get_clear_query() -> str:
    """Get a clear query from the user by checking for ambiguity."""
    classifier = QueryClassifier()
    
    while True:
        query = input("\nEnter your question: ").strip()
        if not query:
            print("Please enter a question.")
            continue
            
        # Check if query is ambiguous
        classifications = classifier.classify(query)
        if classifications.get("ambiguous", 0) > 0.5:
            print("\nYour query seems ambiguous. Could you please be more specific?")
            continue
            
        return query

def extract_content_from_crew_output(crew_output) -> str:
    """Extract the raw content from a CrewOutput object."""
    if hasattr(crew_output, 'raw_output'):
        return crew_output.raw_output
    elif hasattr(crew_output, 'raw'):
        return crew_output.raw
    else:
        return str(crew_output)

def get_retrieval_context(filter_output: str) -> str:
    try:
        parsed = json.loads(filter_output)
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], str) and "\n" not in parsed[0]:
                # Split on periods if it's just one mega string
                return "\n".join(parsed[0].split(". "))
            return "\n".join(parsed)
    except json.JSONDecodeError:
        pass

    # Fall back: try to recover quoted segments manually
    if filter_output.startswith("[") and filter_output.endswith("]"):
        parts = [p.strip() for p in filter_output[1:-1].split('"') if p.strip()]
        return "\n".join(parts)
    
    return filter_output.strip()

def main():
    # Initialize storage components
    qdrant = QdrantStorage()
    graph = GraphStorage()
    keyword_index = KeywordIndex()
    rewriter = QueryRewriter()
    
    # Get query from user
    query = get_clear_query()
    
    # Rewrite query for better search
    rewritten_query = rewriter.rewrite(query)
    print(f"\nRewritten query: {rewritten_query}")
    
    # Get search results
    print("\nRetrieving search results...")
    search_results = get_formatted_results(qdrant, graph, keyword_index, rewritten_query)
    
    # Initialize agents
    llm = get_llm()
    filter_agent = get_filter_agent(llm)
    generator_agent = get_generator_agent(llm)
    
    # Create tasks
    tasks = get_tasks(query, {"filter": filter_agent, "generator": generator_agent}, search_results)
    
    # Create crew with filter and generator tasks
    crew = Crew(
        agents=[filter_agent, generator_agent],
        tasks=tasks,
        verbose=True
    )
    
    # Run the crew
    result = crew.kickoff()
    
    # Extract the raw content from the result
    final_answer = extract_content_from_crew_output(result)
    
    # Print the final answer
    print("\nFinal Answer:")
    print("=" * 50)
    print(final_answer)
    print("=" * 50)
    
    # Evaluate the response if enabled
    if ENABLE_EVALUATION:
        print("\nEvaluating response...")
        filter_task = next(task for task in tasks if task.agent == filter_agent)
        raw_filter_output = extract_content_from_crew_output(filter_task.output)
        
        retrieval_context = get_retrieval_context(raw_filter_output)
        print("\nRetrieval Context:")
        print("=" * 50)
        print(retrieval_context)
        print("=" * 50)
        
        if retrieval_context:
            evaluation = evaluate_response(query, final_answer, retrieval_context, llm)
            print("\nEvaluation Results:")
            print("=" * 50)
            print(json.dumps(evaluation, indent=2))
            print("=" * 50)
        else:
            print("\nWarning: Could not get retrieval context for evaluation")
    

if __name__ == "__main__":
    main()
