"""Tool registry: registration, arg validation, authorization, execution.

The registry is domain-agnostic. Vertical tools (e-commerce) are registered into
it from app/tools/. Two security invariants live here:

  1. The model supplies only tool *arguments* (e.g. an order_number). It never
     supplies a workspace id — the workspace comes from the trusted ToolContext,
     resolved from the caller's API key. So a tool can only ever touch the
     caller's own data.
  2. ``authorize`` runs BEFORE ``execute`` (enforced by the engine). It validates
     argument shape and runs the tool's optional ownership/policy check, raising
     ToolError to block execution.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:  # avoid importing models at engine import time
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.agent.providers.base import EmitFn
    from app.models import Conversation, Workspace


class ToolError(Exception):
    """Recoverable tool failure — fed back to the model as a structured error,
    never surfaced as a 500."""


@dataclass
class ToolContext:
    """Trusted execution context handed to tool handlers.

    ``session_factory`` (not a shared session) lets each tool open its own
    session, so the engine can run tools concurrently without sharing a session.
    """

    workspace: "Workspace"
    conversation: "Conversation"
    session_factory: "async_sessionmaker"
    # Optional SSE emitter so vertical tools can emit domain events (e.g.
    # approval_required) without the engine knowing about them. None when not streaming.
    emit: "EmitFn | None" = None


Handler = Callable[[BaseModel, ToolContext], Awaitable[dict[str, Any]]]
Authorizer = Callable[[BaseModel, ToolContext], Awaitable[None]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler
    authorizer: Authorizer | None = None

    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"Unknown tool: {name}")
        return spec

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions for the provider (name/description/input_schema)."""
        return [
            {"name": s.name, "description": s.description, "input_schema": s.input_schema()}
            for s in self._tools.values()
        ]

    def public_list(self) -> list[dict[str, Any]]:
        """Tool list for GET /v1/tools (parameters_schema naming per the API contract)."""
        return [
            {"name": s.name, "description": s.description, "parameters_schema": s.input_schema()}
            for s in self._tools.values()
        ]

    def _validate(self, spec: ToolSpec, raw: dict[str, Any]) -> BaseModel:
        try:
            return spec.args_model.model_validate(raw or {})
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for '{spec.name}': {exc.errors()}") from exc

    async def authorize(self, name: str, raw_arguments: dict[str, Any], ctx: ToolContext) -> BaseModel:
        """Validate args + run the tool's ownership/policy check. Returns parsed args."""
        spec = self.get(name)
        args = self._validate(spec, raw_arguments)
        if spec.authorizer is not None:
            await spec.authorizer(args, ctx)
        return args

    async def execute(self, name: str, args: BaseModel, ctx: ToolContext) -> dict[str, Any]:
        spec = self.get(name)
        return await spec.handler(args, ctx)
