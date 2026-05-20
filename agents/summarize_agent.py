from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings


SUMMARIZE_SYSTEM = """You are a financial document summarizer.
Distill the provided research into a structured summary with:
1. Key findings (3-5 bullet points)
2. Risk factors identified
3. Quantitative metrics mentioned
Keep language precise and suitable for a quant portfolio manager."""


class SummarizeAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )

    async def run(self, research: str) -> str:
        messages = [
            SystemMessage(content=SUMMARIZE_SYSTEM),
            HumanMessage(content=f"Research to summarize:\n\n{research}"),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content
