import uuid
import json
import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from collections import deque

# -------------------------------------
# Logging setup
# -------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrated-agent")

app = FastAPI(title="SpringBoot Orchestrated Agent")

# -------------------------------------
# In-memory session store (per user session)
# -------------------------------------
session_memory: Dict[str, deque] = {}  # session_id → deque of messages
MAX_HISTORY = 20  # keep more history to preserve tool call context

# -------------------------------------
# Models
# -------------------------------------
class AskRequest(BaseModel):
    query: str
    session_id: str
    tools: Optional[List[dict]] = None
    tool_results: Optional[List[dict]] = None  # Sent later by orchestrator


# -------------------------------------
# Helper: Maintain conversation memory
# -------------------------------------
def update_session_memory(session_id: str, message: dict):
    """Store full message objects (not just role+content)"""
    if session_id not in session_memory:
        session_memory[session_id] = deque(maxlen=MAX_HISTORY)
    session_memory[session_id].append(message)


def get_session_history(session_id: str) -> List[dict]:
    return list(session_memory.get(session_id, []))


# -------------------------------------
# FastAPI Endpoint
# -------------------------------------
@app.post("/ask")
async def ask_agent(
    req: AskRequest,
    openai_api_key: str = Header(..., alias="openai-api-key")
):
    client = AsyncOpenAI(api_key=openai_api_key)

    # --- Case 1️⃣ Initial tool decision ---
    if req.tool_results is None:
        if not req.tools or len(req.tools) == 0:
            raise HTTPException(status_code=400, detail="No tools provided.")

        # Add user message to history
        update_session_memory(req.session_id, {"role": "user", "content": req.query})
        history = get_session_history(req.session_id)

        logger.info(f"🧠 Query: {req.query} | Session: {req.session_id}")

        decision = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                You are an autonomous agent designed to solve user requests by breaking them down into a sequence of steps.
 
*Your Goal:* To fully address the user's request by thinking step-by-step and using the available tools.
 
*Your Process:*
 
1.  *Analyze:* Carefully analyze the user's request, the context, and the conversation history. Formulate a plan or the next immediate step.
2.  *Act:* If a tool can help you proceed, call it. You can call multiple tools in parallel if needed.
3.  *Observe:* Analyze the results from the tool calls to inform your next step.
4.  *Repeat:* Continue this Analyze-Act-Observe cycle until you have gathered all the information needed to provide a complete and final answer.
5.  *Final Answer:* Once you are certain you have completed the request, provide a comprehensive final answer to the user. Do not call any more tools at this stage.
 
You cannot access files or send emails directly. You MUST use the provided tools for these actions.
                """},
                *history
            ],
            tools=req.tools,
            tool_choice="auto",
        )

        choice = decision.choices[0].message

        # --- No tool case ---
        if not getattr(choice, "tool_calls", None):
            response_text = choice.content or "No tool needed for this query."
            
            # Store assistant response
            update_session_memory(req.session_id, {
                "role": "assistant",
                "content": response_text
            })
            
            return {
                "status": "completed",
                "session_id": req.session_id,
                "query": req.query,
                "response": response_text,
                "tools_used": []
            }

        # --- Tool call case ---
        # IMPORTANT: Store the full assistant message with tool_calls
        assistant_message = {
            "role": "assistant",
            "content": choice.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in choice.tool_calls
            ]
        }
        update_session_memory(req.session_id, assistant_message)

        # Build response
        tool_calls = []
        for call in choice.tool_calls:
            logger.info(f'tool call: {call}')
            args = json.loads(call.function.arguments or "{}")
            tool_calls.append({
                "tool_call_id": call.id,
                "params": {
                    "name": call.function.name,
                    "arguments": args
                }
            })

        return {
            "status": "tool_calls_pending",
            "session_id": req.session_id,
            "query": req.query,
            "tools_used": [c.function.name for c in choice.tool_calls],
            "tool_call_plan": tool_calls
        }

    # --- Case 2️⃣ Tool results received ---
    else:
        logger.info(f"🪄 Tool results received for session: {req.session_id}")

        # Get existing history (which includes the assistant message with tool_calls)
        history = get_session_history(req.session_id)

        # Build tool messages for all tool results
        tool_messages = []
        for tool_result in req.tool_results:
            tool_call_id = tool_result.get("tool_call_id")
            if not tool_call_id:
                raise HTTPException(status_code=400, detail="tool_call_id missing in tool_results")

            # Extract content - handle both direct content and nested [{type: "text", text: "..."}] format
            content = tool_result.get("content")
            if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                # Handle [{type: "text", text: "..."}] format
                if "text" in content[0]:
                    tool_output_raw = content[0]["text"]
                else:
                    tool_output_raw = json.dumps(content, indent=2)
            else:
                # Handle direct content format
                tool_output_raw = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
            
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_output_raw
            }
            tool_messages.append(tool_message)
            
            # Store tool result in history
            update_session_memory(req.session_id, tool_message)

        # Get updated history
        history = get_session_history(req.session_id)

        # Ask model how to proceed next
        result = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an autonomous agent.
                    You may:
                    - produce final answer OR
                    - request another tool call.
                    Think step-by-step.
                    """
                },
                *history
            ],
            tools=req.tools,
            tool_choice="auto"
        )

        choice = result.choices[0].message

        # ------------------------------------------------------------------
        # CASE A: No more tool calls → final answer
        # ------------------------------------------------------------------
        if not getattr(choice, "tool_calls", None):
            response_text = choice.content or ""
            
            # Store final response
            update_session_memory(req.session_id, {
                "role": "assistant",
                "content": response_text
            })
            
            return {
                "status": "completed",
                "session_id": req.session_id,
                "query": req.query,
                "response": response_text,
                "raw_tool_results": req.tool_results
            }

        # ------------------------------------------------------------------
        # CASE B: More tool calls → return new tool_call_plan
        # ------------------------------------------------------------------
        # Store assistant message with new tool calls
        assistant_message = {
            "role": "assistant",
            "content": choice.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in choice.tool_calls
            ]
        }
        update_session_memory(req.session_id, assistant_message)

        tool_calls = []
        for call in choice.tool_calls:
            tool_calls.append({
                "tool_call_id": call.id,
                "params": {
                    "name": call.function.name,
                    "arguments": json.loads(call.function.arguments or "{}")
                }
            })

        return {
            "status": "tool_calls_pending",
            "session_id": req.session_id,
            "query": req.query,
            "response": choice.content or "",
            "tool_call_plan": tool_calls,
            "tools_used": [c.function.name for c in choice.tool_calls],
        }
