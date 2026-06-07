import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGroq(model="llama-3.3-70b-versatile")

def process_message(state: AgentState) -> AgentState:
    """ This node is answering the user query and appending the response to the message list."""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print(f" Current State: {state['messages']}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process_message", process_message)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)
agent = graph.compile()

# initialize the empty conversation history
conversation_history = []
# start the conversation loop
user_input = input("User: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("User: ")

# Save the conversation history to a file
with open("conversation_history.txt", "w") as f:
    f.write("your conversation history:\n")
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            f.write(f"User: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            f.write(f"AI: {msg.content}\n")
print("Conversation history saved to conversation_history.txt")