import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
from fastapi import Header, HTTPException, status

from agent.agent import graph

logger = logging.getLogger("backend")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Incoming chat request schema for the agent API."""
    query: Optional[str] = None
    session_id: Optional[str] = None
    tools: List[Dict]
    tool_results: Optional[List[Dict]] = None


class ChatResponse(BaseModel):
    """Response schema returned by the chat endpoint."""
    status: str
    session_id: str
    tools_used: List[str]
    tool_call_plan: Optional[List[Dict]] = None
    response: Optional[str] = None


@app.post("/chat")
async def chat(request: ChatRequest, openai_api_key: str = Header(None, convert_underscores=False, alias="openai_api_key")):
    """
    Primary chat endpoint for interacting with the agent.

    - Accepts user messages, tool definitions, and tool results.
    - Invokes the LangGraph state machine using the provided session_id.
    - Returns either:
        1. A final LLM response, or
        2. Tool call instructions for the client to execute.

    Args:
        request (ChatRequest): The incoming POST body.

    Returns:
        ChatResponse: The completed response or tool call directives.
    """

    logger.info("Received /chat request.")
    logger.debug(f"Raw request body: {request.model_dump()}")

    if request.query is not None:
        logger.info("Constructing new state with user query.")
        state = {
            "query": request.query,
            "messages": [{"role": "user", "content": request.query}],
            "external_tools": request.tools,
            "tool_results": request.tool_results
        }
    else:
        logger.info("Constructing state without user message (tool result follow-up).")
        state = {
            "query": request.query,
            "external_tools": request.tools,
            "tool_results": request.tool_results
        }

    logger.debug(f"Final constructed state: {state}")

    config = {"configurable": {"thread_id": request.session_id, "api_key" : f"{openai_api_key}"}}
    logger.info(f"Invoking LangGraph for session: {request.session_id}")

    try:
        results = graph.invoke(state, config)
        logger.info("Graph invocation completed.")
        logger.debug(f"Graph returned: {results}")


        if results.get('response') is not None:
            logger.info("Returning final LLM response to client.")
            return ChatResponse(
                status="completed",
                session_id=request.session_id,
                response=results['response'],
                tools_used=[]
            )


        logger.info("Returning pending tool call plan to client.")
        return ChatResponse(
            status="tool_calls_pending",
            session_id=request.session_id,
            tools_used=results["tools_used"],
            tool_call_plan=results["tool_call_plan"],
        )

    except Exception as e:
        logger.exception("Error occurred while processing /chat request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
