import os
from dataclasses import dataclass


@dataclass
class Settings:
    db_url: str = os.getenv(
        "PGVECTOR_DB_URL",
        "postgresql+psycopg://ai:ai@localhost:5532/ai",
    )
    table_name: str = os.getenv("PGVECTOR_TABLE_NAME", "legal_docs")
    model_provider: str = os.getenv("MODEL_PROVIDER", "openai").lower()
    legal_manual_url: str = os.getenv(
        "LEGAL_MANUAL_URL",
        "https://www.justice.gov/d9/criminal-ccips/legacy/2015/01/14/ccmanual_0.pdf",
    )
    load_legal_kb: bool = os.getenv("LOAD_LEGAL_KB", "true").lower() == "true"


def get_settings() -> Settings:
    return Settings()
