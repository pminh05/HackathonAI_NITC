"""Category registration interfaces without category-specific rules."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class CategoryDefinition(BaseModel):
    """Location metadata for an independently implemented category."""

    name: str
    package_path: str
    config_path: Path


class CategoryRegistry:
    """In-memory registry for category modules."""

    def __init__(self) -> None:
        self._categories: dict[str, CategoryDefinition] = {}

    def register(self, definition: CategoryDefinition) -> None:
        """Register or replace one category definition."""
        self._categories[definition.name] = definition

    def get(self, name: str) -> CategoryDefinition:
        """Return a category definition by name."""
        return self._categories[name]

    def all(self) -> dict[str, CategoryDefinition]:
        """Return a shallow copy of registered category definitions."""
        return self._categories.copy()


def build_default_registry() -> CategoryRegistry:
    """Register the only implemented MVP category."""
    registry = CategoryRegistry()
    registry.register(
        CategoryDefinition(
            name="refrigerator",
            package_path="advisor.categories.refrigerator",
            config_path=Path(__file__).with_name("refrigerator") / "config.yaml",
        )
    )
    return registry
