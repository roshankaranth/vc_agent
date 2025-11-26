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
from langchain_core.messages import SystemMessage, HumanMessage
from agent.prompt import SYSTEM_PROMPT

LLM_MODEL = "gpt-4o-mini-2024-07-18"
load_dotenv()


class AgentState(MessagesState):
    query: str
    response: str

tools = [fetch_latest_email, retriever, send_email, web_search, web_scrap]


def reasoning_node(state: AgentState):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=1)
    llm_with_tools = llm.bind_tools(tools)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if "messages" in state and state["messages"]:
        messages += state["messages"]

    messages.append(HumanMessage(content=state["query"]))
    response = llm_with_tools.invoke(messages)
    return {"response": response.content, "messages": response}
    

builder = StateGraph(AgentState)
memory = MemorySaver()


builder.add_node("reasoning_node", reasoning_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "reasoning_node")
builder.add_conditional_edges(
    "reasoning_node",
    tools_condition,      
    {
        "tools": "tools",
        "__end__": END,
    },
)
builder.add_edge("tools", "reasoning_node")


graph = builder.compile(checkpointer=memory)