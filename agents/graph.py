"""
LangGraph multi-agent orchestration graph.
Flow: retrieve → research → summarize → done
"""
import asyncio
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langsmith import traceable

from agents.research_agent import ResearchAgent
from agents.summarize_agent import SummarizeAgent


class AgentState(TypedDict):
    query: str
    context_chunks: list[dict]
    research: str
    summary: str
    messages: Annotated[list[str], operator.add]


research_agent = ResearchAgent()
summarize_agent = SummarizeAgent()


@traceable(name="research_node")
async def research_node(state: AgentState) -> AgentState:
    research = await research_agent.run(state["query"], state["context_chunks"])
    return {**state, "research": research, "messages": ["research_complete"]}


@traceable(name="summarize_node")
async def summarize_node(state: AgentState) -> AgentState:
    summary = await summarize_agent.run(state["research"])
    return {**state, "summary": summary, "messages": ["summarize_complete"]}


def should_summarize(state: AgentState) -> str:
    return "summarize" if state.get("research") else END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", should_summarize, {"summarize": "summarize", END: END})
    graph.add_edge("summarize", END)

    return graph.compile()


compiled_graph = build_graph()


async def run_pipeline(query: str, context_chunks: list[dict]) -> dict:
    initial_state: AgentState = {
        "query": query,
        "context_chunks": context_chunks,
        "research": "",
        "summary": "",
        "messages": [],
    }
    result = await compiled_graph.ainvoke(initial_state)
    return {"research": result["research"], "summary": result["summary"]}
