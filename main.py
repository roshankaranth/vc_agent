from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, status


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
    message: str
    session_id : Optional[str] = None

class ChatResponse(BaseModel):
    response : str
    session_id : Optional[str] = None



@app.post("/chat")
async def chat(request: ChatRequest):
    state = {
        "query": request.message,
        "messages": [{"role": "user", "content": request.message}]
    }
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        result = graph.invoke(state, config)
        return ChatResponse(
            response=result["response"],
            session_id=request.session_id
        )

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )