import os

from agno.models.openai import OpenAIChat
from agno.models.google import Gemini

from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.google import GeminiEmbedder

from .config import get_settings


def build_llm():
    cfg = get_settings()

    if cfg.model_provider == "gemini":
        return Gemini(
            id=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        )

    # default: OpenAI
    return OpenAIChat(
        id=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )


def build_embedder():
    cfg = get_settings()

    if cfg.model_provider == "gemini":
        return GeminiEmbedder(
            id=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        )

    # default: OpenAI
    return OpenAIEmbedder(
        id=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    )
