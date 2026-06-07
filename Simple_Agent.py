from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq          # 👈 changed
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatGroq(model="llama-3.3-70b-versatile")  # 👈 changed

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"\nAI Response: {response.content}\n")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    agent.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")