import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY enviorment varaible not set.")

try:
    tavily_client = TavilyClient(TAVILY_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize API clients : {e}")
    raise


def web_search(query : str) -> List[Dict[str, Any]]:
    """
    Performs a web search for a given query.
    Args:
        query: The user's input query

    Return:
        search results.
    """

    search_result = []

    try:
        search_result  = tavily_client.search(
        query = query,
        topic = "general",
        search_depth = "advanced",
        chunks_per_source=3,
        max_results = 5,
        include_raw_content = False,
    )

    except Exception as e:
        logging.error(f"Search failed for query '{query}' : {e}")


    if not search_result:
        logging.info("No search reult found for the query")
        return []
    

    logging.info(f"Returning {len(search_result)} final results from web search.")
    return search_result

