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


def web_scrap(url : str) -> List[Dict[str, Any]]:
    """
    Performs a web scrap for a given url
    Args:
        url: The user's input url

    Return:
        search results.
    """

    try:
        search_result  = tavily_client.extract(
            urls=url,
            extract_dept = "advanced",
            include_images=False,
            include_favicon=False
            )

    except Exception as e:
        logging.error(f"Extraction failed for url '{url}' : {e}")


    if not search_result:
        logging.info("No search result found for the url")
        return []
    

    logging.info(f"Returning {len(search_result)} final results from web extract.")
    return search_result

