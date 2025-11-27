from langgraph.graph import MessagesState
from typing import List, Dict

tools = [
   {
    "type": "function",
    "function": {
      "name": "web_search",
      "description": "Performs a web search using the Tavily search client and returns the top relevant results.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "A natural language search query to look up on the web."
          }
        },
        "required": ["query"]
      }
    }
  },
  {
  "type": "function",
  "function": {
    "name": "rag_retrieve",
    "description": "Retrieves and ranks documents relevant to the user query using vector search.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_query": {
          "type": "string",
          "description": "The user query from which to retrieve relevant documents."
        }
      },
      "required": ["user_query"]
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "web_scrap",
    "description": "Scrapes a webpage using Tavily extract and returns structured extraction data.",
    "parameters": {
      "type": "object",
      "properties": {
        "url": {
          "type": "string",
          "description": "The URL of the webpage to scrape."
        }
      },
      "required": ["url"]
    }
  }
},
]

Internal_Tools = ["web_search", "web_scrap", "rag_retrieve"]

class AgentState(MessagesState):
    query: str
    response: str
    external_tools : List[Dict]
    tool_results : List[Dict]
    tool_call_plan : List[Dict]
    tools_used : List[str]

    