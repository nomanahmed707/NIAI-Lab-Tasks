from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

DOCUMENT_CONTENT = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update_document_content(new_content: str) -> str:
    """Tool to update the document content."""
    global DOCUMENT_CONTENT
    DOCUMENT_CONTENT = new_content
    return "Document content updated. Current content is:\n" + DOCUMENT_CONTENT


@tool
def save_document(file_name: str) -> str:
    """Tool to save the document content to a file.
    Args:
        file_name: The name of the file to save the document content to.
    """
    global DOCUMENT_CONTENT
    if not file_name.endswith('.txt'):
        file_name += '.txt'
    try:
        with open(file_name, 'w') as file:
            file.write(DOCUMENT_CONTENT)
        return f"Document saved successfully to '{file_name}'."
    except Exception as e:
        return f"Failed to save document: {str(e)}"


tools = [update_document_content, save_document]
llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)


def main_agent(state: AgentState) -> AgentState:
    system_message = SystemMessage(content=f"""You are Drafter, a helpful writing assistant.
    
    - If the user wants to update or modify content, use the 'update_document_content' tool.
    - If the user wants to save and finish, use the 'save_document' tool.
    - Always show the current document state after modifications.
    
    Current document content: {DOCUMENT_CONTENT}""")

    if not state['messages']:
        # First turn — no user input yet
        user_message = HumanMessage(content="Hello, I need help updating a document.")
        all_messages = [system_message, user_message]
        response = llm.invoke(all_messages)
        print(f"\n🤖 AI: {response.content}")
        return {'messages': [user_message, response]}  # ✅ fixed — returns first turn

    else:
        # Get new user input
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
        all_messages = [system_message] + list(state['messages']) + [user_message]
        response = llm.invoke(all_messages)

        print(f"\n🤖 AI: {response.content}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

        return {'messages': list(state['messages']) + [user_message, response]}


def should_continue(state: AgentState) -> str:  # ✅ returns string not bool
    messages = state['messages']
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved successfully" in message.content.lower()):
            return "end"
    return "continue"


def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("Agent", main_agent)
graph.add_node("tool_node", ToolNode(tools=tools))

graph.set_entry_point("Agent")
graph.add_edge("Agent", "tool_node")

graph.add_conditional_edges(          # ✅ fixed format
    "tool_node",
    should_continue,
    {
        "continue": "Agent",
        "end": END,
    }
)

app = graph.compile()


def run_agent():
    print("\n===== DRAFTER =====")
    state = {'messages': []}
    for step in app.stream(state, stream_mode="values"):  # ✅ stream not app()
        if "messages" in step:
            print_messages(step["messages"])
    print("\n===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_agent()