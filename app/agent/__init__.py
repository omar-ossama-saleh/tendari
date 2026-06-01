"""Domain-agnostic agent engine.

Nothing in this package may contain vertical (e-commerce) logic — that lives in
app/tools/. The engine only knows about a Provider, a ToolRegistry, and
conversation/message persistence.
"""
