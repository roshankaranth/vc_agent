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

]