import os
import asyncio

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.google import GeminiEmbedder


# ---------- Config ----------

DB_URL = os.getenv("PGVECTOR_DB_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
TABLE_NAME = os.getenv("PGVECTOR_TABLE_NAME", "legal_docs")

# "openai" or "gemini"
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()

LEGAL_MANUAL_URL = os.getenv(
    "LEGAL_MANUAL_URL",
    "https://www.justice.gov/d9/criminal-ccips/legacy/2015/01/14/ccmanual_0.pdf",
)


def build_embedder():
    """Choose the embedding model based on provider (OpenAI or Gemini)."""
    if MODEL_PROVIDER == "gemini":
        return GeminiEmbedder(
            id=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        )
    else:
        # Default: OpenAI
        return OpenAIEmbedder(
            id=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        )


def build_model():
    """Choose the LLM based on provider."""
    if MODEL_PROVIDER == "gemini":
        return Gemini(
            id=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        )
    else:
        # Default: GPT
        return OpenAIChat(
            id=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )


async def load_knowledge_async(knowledge: Knowledge) -> None:
    """
    Load the DOJ cybercrime manual into the knowledge base.

    Run once with LOAD_LEGAL_KB=true to populate the DB,
    then you can set it to false for faster startup.
    """
    await knowledge.add_content_async(
        url=LEGAL_MANUAL_URL,
        metadata={
            "source": "DOJ Computer Crime & Intellectual Property Section Manual",
        },
    )


def main():
    # 1) Build vector DB with the right embedder
    vector_db = PgVector(
        table_name=TABLE_NAME,
        db_url=DB_URL,
        embedder=build_embedder(),
    )

    # 2) Attach it to Knowledge
    knowledge = Knowledge(vector_db=vector_db)

    # 3) (Optional) Load / refresh knowledge from the PDF URL
    #    After first successful run, set LOAD_LEGAL_KB=false to avoid re-embedding.
    if os.getenv("LOAD_LEGAL_KB", "true").lower() == "true":
        asyncio.run(load_knowledge_async(knowledge))

    # 4) Build the legal advisor agent
    legal_agent = Agent(
        name="LegalAdvisor",
        knowledge=knowledge,
        search_knowledge=True,
        model=build_model(),
        markdown=True,
        instructions=[
            "Provide legal information and analysis based on the knowledge base.",
            "Ground your answers in specific passages from the manual whenever possible.",
            "Include relevant citations (sections, page numbers, statutes) when you can.",
            "Always clarify that you are providing general legal information, "
            "not professional legal advice.",
            "Recommend consulting with a licensed attorney for any specific legal situation.",
        ],
    )

    # 5) Example query
    question = (
        "What are the legal consequences and criminal penalties for spoofing an email address?"
    )

    legal_agent.print_response(question, stream=True)


if __name__ == "__main__":
    main()
