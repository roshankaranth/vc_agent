from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from agent.tools.mail_tool import fetch_latest_email, send_email
from agent.tools.retriever import retriever
from agent.tools.web_search import web_search
from agent.tools.web_scraping import web_scrap
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from agent.prompt import SYSTEM_PROMPT
from openai import OpenAI
from agent.state import tools
from typing import List, Dict
import json
from langchain_core.messages.tool import ToolCall

LLM_MODEL = "gpt-4o-mini-2024-07-18"
load_dotenv()

def convert_msg_to_dict(msg):
    """Convert LangChain messages → OpenAI Messages API format."""
    if msg.type == "system":
        return {"role": "system", "content": msg.content}
    if msg.type == "human":
        return {"role": "user", "content": msg.content}
    if msg.type == "ai":
        openai_tool_calls = []
        if msg.tool_calls:
            for tool in msg.tool_calls:
                openai_tool_calls += [{'id' : tool['id'], 'function' : {'arguments' : json.dumps(tool['args']), 'name' : tool['name']}, 'type' : 'function'}]
            return {"role": "assistant", "content": msg.content, "tool_calls" : openai_tool_calls}
        return {"role": "assistant", "content": msg.content}
    if msg.type == "tool":
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id
        }
    raise ValueError(f"Unknown message type: {msg}")


def from_openai_msg(msg):
    """Convert *dict* from OpenAI ChatCompletion → LangChain Message."""
    role = msg["role"]
    content = '' if msg.get("content") is None else msg['content']

    lc_tool_calls = []
    if msg.get('tool_calls') is not None:
        for tool in msg['tool_calls']:
            tool_id = tool["id"]
            tool_name = tool["function"]["name"]
            tool_args = json.loads(tool["function"]["arguments"])

            lc_tool_calls += [ToolCall(name=tool_name, args=tool_args, id=tool_id)]

    return AIMessage(content=content, tool_calls = lc_tool_calls)


class AgentState(MessagesState):
    query: str
    response: str
    external_tools : List[Dict]
    tool_results : List[Dict]
    tool_call_plan : List[Dict]
    tools_used : List[str]


def reasoning_node(state: AgentState):
    client = OpenAI()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    openai_messages = []
    tool_messages = []
    if state.get("messages"):
        messages += state["messages"]

    if state.get("tool_results"):
        for t in state["tool_results"]:
            tool_messages.append(
                ToolMessage(
                    content=t["content"],
                    tool_call_id=t["tool_call_id"]
                )
            )
        messages.extend(tool_messages)

    for msg in messages:
        openai_messages = openai_messages + [convert_msg_to_dict(msg)]
    runtime_tools = state["external_tools"]

    decision = client.chat.completions.create(
        model=LLM_MODEL,
        messages=openai_messages,
        tools=runtime_tools,
        tool_choice="auto",
    )

    choice = decision.choices[0].message
    lc_messages = from_openai_msg(choice.model_dump())
    lc_messages = tool_messages + [lc_messages]

    if choice.tool_calls is None:
        response_text = choice.content or ""
        return {
            "response": response_text,
            "messages": lc_messages,  
        }
    else:
        tool_calls = []
        for call in choice.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            tool_calls.append({
                "tool_call_id": call.id,
                "params": {
                    "name": call.function.name,
                    "arguments": args
                }
            })

        return {
            "tools_used": [c.function.name for c in choice.tool_calls],
            "tool_call_plan": tool_calls,
            "messages": lc_messages,
            "response" : None
        }


builder = StateGraph(AgentState)
memory = MemorySaver()

builder.add_node("reasoning_node", reasoning_node)
# builder.add_node("tool_node", tool_node)


builder.add_edge(START, "reasoning_node")
# builder.add_conditional_edges(
#     "reasoning_node",
#     tools_condition,      
#     {
#         "tools": "tool_node",
#         "__end__": END,
#     },
# )
# builder.add_edge("tools", "reasoning_node")
builder.add_edge("reasoning_node", END)


graph = builder.compile(checkpointer=memory)