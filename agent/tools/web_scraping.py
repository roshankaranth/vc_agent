import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from tavily import TavilyClient
import re

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY enviorment varaible not set.")

try:
    tavily_client = TavilyClient(TAVILY_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize API clients : {e}")
    raise

def clean_webpage_text(text: str) -> str:
    """
    Cleans raw webpage text extracted from Tavily by removing:
    - duplicate lines
    - navigation menus
    - ads or repeated promotional content
    - lines with only symbols or emojis
    - image alt-text artifacts
    - extra whitespace

    Returns a cleaner, human-readable string.
    """

    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  

    patterns_to_remove = [
        r"Black Friday sale is live now.*", 
        r"Use this code.*",
        r"Try Bitscale Now.*",
        r"Need more information.*",
        r"Back to Directory.*"
    ]

    for p in patterns_to_remove:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    text = re.sub(r'[^\w\s.,:/\-()]+', '', text)

    lines = text.split("\n")

    cleaned_lines = []
    seen = set()

    for line in lines:
        line = line.strip()

        if not line or len(line) < 3:
            continue

        if line.isupper() and len(line.split()) <= 3:
            continue

        if line in seen:
            continue

        seen.add(line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def web_scrap(url : str) -> List[Dict[str, Any]]:
    """
    Performs a web scrap for a given url
    Args:
        url: The user's input url

    Return:
        search results.
    """

    clean_results = []

    try:
        extraction  = tavily_client.extract(
            urls=url,
            extract_dept = "advanced",
            include_images=False,
            include_favicon=False
            )
        
        raw = extraction["results"][0].get("raw_content", "")
        
        try:
            clean_results = clean_webpage_text(raw)
        except Exception as e:
            clean_results = extraction

    except Exception as e:
        logging.error(f"Extraction failed for url '{url}' : {e}")

    if not clean_results:
        logging.info("No search result found for the url")
        return []
    
    logging.info(f"Returning {len(clean_results)} final results from web extract.")
    return clean_results

