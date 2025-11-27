import json
import logging
from dotenv import load_dotenv
from agent.prompt import SYSTEM_PROMPT
from openai import OpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages.tool import ToolCall
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage

from agent.state import tools, Internal_Tools, AgentState
from agent.tools.retriever import retriever
from agent.tools.web_search import web_search
from agent.tools.web_scraping import web_scrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

LLM_MODEL = "gpt-4o-mini-2024-07-18"

TOOL_REGISTRY = {
    "web_search": web_search,
    "web_scrap": web_scrap,
    "rag_retrieve": retriever.func if hasattr(retriever, "func") else retriever
}

load_dotenv()

def convert_msg_to_dict(msg):
    """
    Convert LangChain Message objects into OpenAI ChatCompletion message dictionaries.

    Args:
        msg: A LangChain message instance.

    Returns:
        dict: A message dictionary formatted for OpenAI's Messages API.
    """
    
    logger.debug(f"Converting LC message to OpenAI format: {msg}")

    if msg.type == "system":
        return {"role": "system", "content": msg.content}

    if msg.type == "human":
        return {"role": "user", "content": msg.content}

    if msg.type == "ai":
        openai_tool_calls = []
        if msg.tool_calls:
            for tool in msg.tool_calls:
                openai_tool_calls.append({
                    'id': tool['id'],
                    'function': {
                        'arguments': json.dumps(tool['args']),
                        'name': tool['name']
                    },
                    'type': 'function'
                })
            return {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": openai_tool_calls
            }

        return {"role": "assistant", "content": msg.content}

    if msg.type == "tool":
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id
        }

    raise ValueError(f"Unknown message type: {msg}")


def from_openai_msg(msg):
    """
    Convert an OpenAI-completion-style message dict back into a LangChain AIMessage.

    Args:
        msg (dict): The assistant message returned by the OpenAI API.

    Returns:
        AIMessage: Converted LangChain-compatible message.
    """
    
    logger.debug(f"Parsing OpenAI message into LangChain AIMessage: {msg}")

    content = '' if msg.get("content") is None else msg['content']
    lc_tool_calls = []

    if msg.get('tool_calls'):
        for tool in msg['tool_calls']:
            tool_id = tool["id"]
            tool_name = tool["function"]["name"]
            tool_args = json.loads(tool["function"]["arguments"])

            lc_tool_calls.append(
                ToolCall(name=tool_name, args=tool_args, id=tool_id)
            )

    return AIMessage(content=content, tool_calls=lc_tool_calls)

def tools_to_description_string(tools: list) -> str:
    """
    Converts a list of external tool definitions (JSON format) into a readable
    string description suitable for system prompts.

    Args:
        tools (list): List of tool metadata dicts matching the OpenAI tool schema.

    Returns:
        str: Human-friendly description of the tools.
    """
    lines = []
    for i, tool in enumerate(tools, start=1):
        fn = tool.get("function", {})
        name = fn.get("name", "unknown_tool")
        desc = fn.get("description", "").strip()
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        lines.append(f"{i}. {name}")
        lines.append(f"   {desc}")

        if not props:
            lines.append("   Parameters: none")
        else:
            lines.append("   Parameters:")
            for p_name, p_info in props.items():
                p_desc = p_info.get("description", "")
                p_type = p_info.get("type", "unknown")
                req_mark = " (required)" if p_name in required else ""
                lines.append(f"     - {p_name} ({p_type}){req_mark}: {p_desc}")

        lines.append("") 

    return "\n".join(lines).strip()

def reasoning_node(state: AgentState, config):
    """
    Perform LLM reasoning. Decide whether to:
    - Respond normally
    - Call internal tools (handled inside graph)
    - Call external tools (execution returned to backend)

    Args:
        state (AgentState): Full graph state.

    Returns:
        Either an LLM response or a Command directing execution to tool_node.
    """
    
    logger.info("Entering reasoning_node.")
    logger.debug(f"Incoming state: {state}")

    openai_messages = []
    tool_messages = []

    external_tool_desc = ''

    try:
        external_tool_desc = "\n\nExternal Tools Attached : \n\n" + tools_to_description_string(state["external_tools"])
        logger.info("External tool description generated sucessfully")

    except Exception as e:
        logger.info(f"Error when generating tool desciption. Fallback to no tool description : {e}")

    messages = [SystemMessage(content=SYSTEM_PROMPT + external_tool_desc)]
    if state.get("messages"):
        messages += state["messages"]

    if state.get("tool_results"):
        logger.info("Adding tool results back into message context.")
        for t in state["tool_results"]:
            tool_messages.append(
                ToolMessage(content=t["content"], tool_call_id=t["tool_call_id"])
            )
        messages.extend(tool_messages)

    for msg in messages:
        openai_messages.append(convert_msg_to_dict(msg))

    runtime_tools = tools + state["external_tools"]
    logger.info(f"Calling LLM with {len(openai_messages)} messages and {len(runtime_tools)} tools.")

    client = OpenAI(api_key=config["configurable"]["api_key"])
    decision = client.chat.completions.create(
        model=LLM_MODEL,
        messages=openai_messages,
        tools=runtime_tools,
        tool_choice="auto",
    )

    choice = decision.choices[0].message
    logger.info("LLM returned a decision.")

    lc_messages = from_openai_msg(choice.model_dump())
    lc_messages = tool_messages + [lc_messages]

    if choice.tool_calls is None:
        logger.info("LLM responded normally (no tool calls).")
        return {
            "response": choice.content or "",
            "messages": lc_messages,
        }

    logger.info("LLM requested tool calls. Classifying internal vs external.")

    internal = []
    external = []

    for call in choice.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        name = call.function.name

        call_plan = {
            "tool_call_id": call.id,
            "params": {"name": name, "arguments": args}
        }

        if name in Internal_Tools:
            internal.append(call_plan)
        else:
            external.append(call_plan)

    if internal and external:
        logger.error("Mixed internal & external tool calls detected!")
        raise ValueError("Mixed internal and external tool calls are not allowed.")

    if internal:
        logger.info(f"Routing {len(internal)} INTERNAL tool calls to tool_node.")
        return Command(
            update={
                "tool_call_plan": internal,
                "tools_used": [plan["params"]["name"] for plan in internal],
                "messages": lc_messages,
            },
            goto="tool_node"
        )

    logger.info(f"Returning {len(external)} EXTERNAL tool calls to backend.")
    return {
        "tools_used": [plan["params"]["name"] for plan in external],
        "tool_call_plan": external,
        "messages": lc_messages,
        "response": None,
    }

def tool_node(state: AgentState):
    """
    Execute INTERNAL tools inside the graph (NOT returned to backend).

    Args:
        state (AgentState): Current agent state containing tool_call_plan.

    Returns:
        dict: Contains tool_results to be fed back into reasoning_node.
    """
    
    logger.info("Entering tool_node for internal tool execution.")
    logger.debug(f"Tool call plan: {state['tool_call_plan']}")

    response = []

    for tool_call in state["tool_call_plan"]:
        name = tool_call["params"]["name"]
        args = tool_call["params"]["arguments"]

        logger.info(f"Executing internal tool: {name} with args {args}")

        if name not in TOOL_REGISTRY:
            logger.error(f"Unknown internal tool requested: {name}")
            raise ValueError(f"Unknown tool: {name}")

        result = TOOL_REGISTRY[name](**args)
        logger.debug(f"Tool result for {name}: {result}")

        tool_result = {
            "content": json.dumps(result),
            "tool_call_id": tool_call["tool_call_id"]
        }
        response.append(tool_result)

    logger.info("Finished executing internal tools.")
    return {
        "tools_used": [],
        "tool_call_plan": [],
        "tool_results": response
    }


builder = StateGraph(AgentState)
memory = MemorySaver()

builder.add_node("reasoning_node", reasoning_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "reasoning_node")
builder.add_edge("tool_node", "reasoning_node")
builder.add_edge("reasoning_node", END)

graph = builder.compile(checkpointer=memory)

logger.info("Agent graph compiled successfully.")
