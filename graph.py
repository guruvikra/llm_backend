# graph.py

from typing import TypedDict, Annotated
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Set up LLM and tools
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
search_tool = TavilySearchResults(max_results=5)
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)

# Memory for checkpointing
memory = MemorySaver()

# Define the graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Model node function
async def init_model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [result]}

# Router to check for tool usage
async def tool_router(state: State):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tool_node"
    return END

# Build the graph
graph = StateGraph(State)
graph.add_node("model", init_model)
graph.add_node("tool_node", ToolNode(tools=tools))
graph.set_entry_point("model")
graph.add_conditional_edges("model", tool_router)
graph.add_edge("tool_node", "model")

compiled_graph = graph.compile(checkpointer=memory)

# Optional: expose reusable config
def get_config():
    return {"configurable": {"thread_id": str(uuid4())}}
