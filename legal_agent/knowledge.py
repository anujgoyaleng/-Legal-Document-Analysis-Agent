from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

from .config import get_settings
from .models import build_embedder


def create_knowledge() -> Knowledge:
    cfg = get_settings()

    vector_db = PgVector(
        table_name=cfg.table_name,
        db_url=cfg.db_url,
        embedder=build_embedder(),
    )

    return Knowledge(vector_db=vector_db)


async def load_legal_manual_async(knowledge: Knowledge) -> None:
    """
    Load the DOJ manual into the vector DB.

    Call once to populate; after that you can skip it for faster startup.
    """
    cfg = get_settings()

    await knowledge.add_content_async(
        url=cfg.legal_manual_url,
        metadata={
            "source": "DOJ Computer Crime & Intellectual Property Section Manual",
        },
    )
