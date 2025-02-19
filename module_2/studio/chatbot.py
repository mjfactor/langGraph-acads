from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults, YouTubeSearchTool
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

SYSTEM_PROMPT = """You are an expert AI Resume Assistant. Your primary role is to help users improve their resumes and career materials by providing clear, accurate, and up-to-date advice. You can search the web to verify facts, incorporate the latest job market trends, and gather additional context when needed. 

When evaluating a resume or answering questions about resume optimization:
- Use verified information from reliable online sources.
- Incorporate current best practices and relevant industry trends.
- Maintain a professional, clear, and supportive tone at all times.
- If uncertain about details, engage your web search capability to confirm and enrich your response.
- When using youtube search, always return 4 results.

Focus on providing action-oriented, constructive feedback that helps users highlight their skills and experience effectively."""
search = TavilySearchResults()
youtube = YouTubeSearchTool()
tools = [search, youtube]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
llm_with_tools = llm.bind_tools(tools)


class State(MessagesState):
    pass


graph_builder = StateGraph(State)


# Node for the chatbot
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "1"}}


# The config is the **second positional argument** to stream() or invoke()!
