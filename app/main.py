from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, status
from typing import List, Dict

from agent.agent import graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: Optional[str] = None
    session_id : Optional[str] = None
    tools : List[Dict]
    tool_results : Optional[List[Dict]] = None

class ChatResponse(BaseModel):
    status : str
    session_id : str
    tools_used : List[str]
    tool_call_plan : Optional[List[Dict]] = None
    response : Optional[str] = None



@app.post("/chat")
async def chat(request: ChatRequest):

    if request.query is not None : 
        state = {
        "query": request.query,
        "messages": [{"role": "user", "content": request.query}],
        "external_tools" : request.tools,
        "tool_results" : request.tool_results
    }
    else:
        state = {
        "query": request.query,
        "external_tools" : request.tools,
        "tool_results" : request.tool_results
    }
    
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        results = graph.invoke(state, config)
        if results.get('response') is not None:
            return ChatResponse(
                status = "completed",
                session_id = request.session_id,
                response=results['response'],
                tools_used=[]
            )
        else:
            return ChatResponse(
                status = "tool_calls_pending",
                session_id = request.session_id,
                tools_used=results["tools_used"],
                tool_call_plan=results["tool_call_plan"],
            )


    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )