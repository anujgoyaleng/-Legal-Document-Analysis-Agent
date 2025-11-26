from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge

from .models import build_llm


def create_legal_agent(knowledge: Knowledge) -> Agent:
    return Agent(
        name="LegalAdvisor",
        knowledge=knowledge,
        search_knowledge=True,
        model=build_llm(),
        markdown=True,
        instructions=[
            "Provide legal information and analysis based on the loaded knowledge base.",
            "Rely primarily on the retrieved sections from the DOJ manual.",
            "Quote or paraphrase relevant passages and include citations "
            "(sections, page numbers, statutes) when possible.",
            "Always clarify that you are providing general legal information, "
            "not professional legal advice.",
            "Recommend consulting with a licensed attorney for any specific case "
            "or jurisdiction-specific issue.",
        ],
    )
