from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings


RESEARCH_SYSTEM = """You are a quantitative financial research analyst.
Given retrieved document excerpts, synthesize the key facts, figures, and risks.
Be concise, cite specific metrics, and flag any data inconsistencies."""


class ResearchAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

    async def run(self, query: str, context_chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c.get('chunk_id', 'unknown')}]\n{c['text']}"
            for c in context_chunks
        )
        messages = [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=f"Query: {query}\n\nContext:\n{context}\n\nProvide a research synthesis:"
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content
