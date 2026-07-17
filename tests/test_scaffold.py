"""Smoke tests for the repository scaffold."""

from __future__ import annotations

from pathlib import Path

from advisor.categories.registry import CategoryDefinition, CategoryRegistry
from advisor.schemas import ApplicationSettings


def test_modules_import() -> None:
    """Core scaffold modules import without initializing external clients."""
    import advisor.graph
    import advisor.memory.mem0
    import advisor.nodes
    import advisor.persistence.checkpointer
    import advisor.retrieval.qdrant


def test_graph_can_be_built() -> None:
    """The graph compiles without Gemini, Qdrant, or persistence configuration."""
    from advisor.graph import build_graph

    assert build_graph() is not None


def test_registry_can_register_category() -> None:
    """A category definition can be registered in the empty registry."""
    registry = CategoryRegistry()
    definition = CategoryDefinition(
        name="example",
        package_path="advisor.categories.example",
        config_path=Path("example/config.yaml"),
    )

    registry.register(definition)

    assert registry.get("example") == definition


def test_settings_read_fake_environment(monkeypatch) -> None:
    """Settings are read from environment variables without external calls."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    settings = ApplicationSettings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.gemini_model == "test-model"
