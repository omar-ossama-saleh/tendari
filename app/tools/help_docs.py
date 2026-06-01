"""Vertical tool: search_help_docs — RAG retrieval over the workspace's docs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.registry import ToolContext, ToolSpec
from app.rag.retrieve import retrieve_chunks


class SearchHelpDocsArgs(BaseModel):
    query: str = Field(
        ...,
        description="A natural-language search query describing what the customer needs.",
        min_length=1,
    )


async def _search_help_docs(args: SearchHelpDocsArgs, ctx: ToolContext) -> dict:
    # Own session (engine may run tools concurrently); scoped to the workspace.
    async with ctx.session_factory() as session:
        chunks = await retrieve_chunks(session, ctx.workspace.id, args.query)
    return {
        "results": [
            {
                "doc_title": c.doc_title,
                "content": c.content,
                "score": round(c.score, 3),
            }
            for c in chunks
        ]
    }


SEARCH_HELP_DOCS = ToolSpec(
    name="search_help_docs",
    description=(
        "Search the store's help documents (policies, FAQs, guides) and return the "
        "most relevant passages with their document titles. Use this BEFORE answering "
        "any policy, returns, shipping, or how-to question. If it returns nothing "
        "relevant, tell the customer the docs don't cover it instead of guessing."
    ),
    args_model=SearchHelpDocsArgs,
    handler=_search_help_docs,
)
