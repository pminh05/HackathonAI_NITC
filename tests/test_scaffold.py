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


def test_default_category_implements_shared_contract() -> None:
    """Every enabled default category can be loaded without external services."""
    from advisor.categories.registry import build_default_registry

    registry = build_default_registry()
    registry.validate_all()

    for name, definition in registry.all().items():
        if not definition.implemented:
            continue
        spec = registry.get_spec(name)
        spec.validate()
        assert spec.name == name
        assert spec.config["category"] == name
        assert callable(spec.build_filter)
        assert callable(spec.normalize_candidate)


def test_settings_read_fake_environment(monkeypatch) -> None:
    """Settings are read from environment variables without external calls."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("FPT_API_KEY", "test-fpt-key")
    monkeypatch.setenv("FPT_MODEL", "test-fpt-model")
    monkeypatch.setenv("FPT_BASE_URL", "https://example.fptcloud.com")
    monkeypatch.setenv("FPT_ENABLE_THINKING", "true")
    monkeypatch.setenv("FPT_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("FPT_MAX_RETRIES", "1")

    settings = ApplicationSettings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.gemini_model == "test-model"
    assert settings.fpt_api_key is not None
    assert settings.fpt_api_key.get_secret_value() == "test-fpt-key"
    assert settings.fpt_model == "test-fpt-model"
    assert settings.fpt_base_url == "https://example.fptcloud.com"
    assert settings.fpt_enable_thinking is True
    assert settings.fpt_timeout_seconds == 12.5
    assert settings.fpt_max_retries == 1


def test_public_frontend_origin_is_included_in_cors() -> None:
    settings = ApplicationSettings(
        _env_file=None,
        api_cors_origins="http://localhost:5173",
        api_public_frontend_url="https://example.vercel.app/",
    )

    assert settings.cors_origins == [
        "http://localhost:5173",
        "https://example.vercel.app",
    ]
