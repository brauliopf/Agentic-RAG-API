from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_tavily import TavilySearch
from .retriever import retrieve_execute_parallel

@tool
def tavily_search_tool(
    query: Annotated[str, 'The query to search Tavily for']
) -> str:
    """Perform a search on Tavily"""
    print(f">>>>> Searching Tavily for: {query}")
    return TavilySearch(max_results=3).run(query)

@tool
async def retrieve_for_user_id(query: str, user_id: Annotated[str, InjectedToolArg]) -> str: # type: ignore
  """Search and return information from a user-specific knowledge base."""
  return await retrieve_execute_parallel(user_id, query)