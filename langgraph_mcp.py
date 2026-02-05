from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, BaseTool
from dotenv import load_dotenv
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
import sqlite3
import requests
import os
import asyncio
import aiosqlite
import threading

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()
_ = load_dotenv(find_dotenv())
groq_api_key = os.environ["GROQ_API_KEY"]
MATH_MCP_PATH = os.getenv("MATH_MCP_PATH")

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or "qwen/qwen3-32b"
    
    api_key=groq_api_key
)

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


FASTMCP_TOKEN = os.getenv("FASTMCP_TOKEN")
client =MultiServerMCPClient(
    {
        "maths-tool":{
            "transport":"stdio",
            "args":[MATH_MCP_PATH] if MATH_MCP_PATH else None,
            "command":"python"
        },

        "expense": {
        "transport": "http",
        "url": "https://querulous-coral-piranha.fastmcp.app/mcp",
        "headers": {
        "Authorization": f"Bearer {FASTMCP_TOKEN}"
        }
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print("⚠ MCP tools unavailable:", e)
        return []
mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]

# base_tools = [search_tool,get_stock_price]
# mcp_tools = load_mcp_tools()
# if mcp_tools:
#     tools = base_tools + mcp_tools
# else:
#     tools = base_tools

llm_with_tools = llm.bind_tools(tools)

llm_with_tools = llm.bind_tools(tools) if tools else llm


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

async def chat_node(state:ChatState):
    messages=state["messages"]
    response=await llm_with_tools.ainvoke(messages)
    return {"messages":[response]}


from langchain_core.messages import ToolMessage
class SafeToolNode(ToolNode):
    async def ainvoke(self, state, config=None):
        result = await super().ainvoke(state, config)

        new_messages = []
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                msg.content = str(msg.content)  
            new_messages.append(msg)

        return {"messages": new_messages}

tool_node = SafeToolNode(tools) if tools else None




async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot_mcp.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())
graph= StateGraph(ChatState)

graph.add_node("chat node",chat_node)
graph.add_edge(START,"chat node")

if tool_node:
    graph.add_node("tools",tool_node)
    graph.add_conditional_edges("chat node",tools_condition)
    graph.add_edge("tools","chat node")
else:
    graph.add_edge("chat node",END)

chatbot = graph.compile(checkpointer=checkpointer)

async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
