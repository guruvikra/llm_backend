from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from uuid import uuid4
import json

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

search_tool = TavilySearchResults(max_results=5)
 
tools = [search_tool]

memory = MemorySaver()

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    """State of the graph."""

    messages: Annotated[list, add_messages]


async def init_model(state: State):
    """Initialize the state."""
    result = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [result]
    }


async def tool_router(state: State):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tool_node"
    else:
        return END
    
tool_node = ToolNode(tools = tools)

graph = StateGraph(State)


graph.add_node("model", init_model)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("model")

graph.add_conditional_edges("model", tool_router)
graph.add_edge("tool_node", "model")

compiled_graph = graph.compile(checkpointer=memory)


thread_id = str(uuid4())  # Thread ID generated once for the session
config = {
    "configurable": {"thread_id": thread_id}
}

async def main():
    print("Start chatting with the assistant (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        # Stream events from the graph
        events = compiled_graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        )

        async for event in events:
        # Log the entire event to inspect the structure if needed
        # print(f"Event: {event}")

        # Check if the event is a tuple
            if isinstance(event, tuple):
                # Extract the first element, which is the actual message
                msg = event[0]

                if isinstance(msg, AIMessage) and msg.content.strip():
                    print(f"Assistant: {msg.content}")
            else:
                print("Unexpected event structure:", event)

import asyncio
# Run the event loop
asyncio.run(main())