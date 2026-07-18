"""Context-managed SQLite persistence for LangGraph checkpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, AsyncIterator, Iterator

from advisor.schemas import ApplicationSettings

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@contextmanager
def open_sqlite_checkpointer(
    settings: ApplicationSettings,
) -> Iterator[SqliteSaver]:
    """Keep the SQLite connection alive for the compiled graph's lifetime."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError("Install SQLite checkpoint dependencies first.") from exc

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
        yield saver


# The old name remains as a clear alias, but now correctly returns a context manager.
create_sqlite_checkpointer = open_sqlite_checkpointer


@asynccontextmanager
async def open_async_sqlite_checkpointer(
    settings: ApplicationSettings,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Open the async saver used by FastAPI for HITL across HTTP requests."""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        raise RuntimeError("Install SQLite checkpoint dependencies first.") from exc

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_db_path)
    ) as saver:
        await saver.setup()
        yield saver


@asynccontextmanager
async def open_async_checkpointer(
    settings: ApplicationSettings,
) -> AsyncIterator[object]:
    """Open the configured durable checkpointer for the API lifetime."""
    if settings.checkpoint_backend == "sqlite":
        async with open_async_sqlite_checkpointer(settings) as saver:
            yield saver
        return

    if settings.supabase_database_url is None:
        raise ValueError(
            "SUPABASE_DATABASE_URL is required when CHECKPOINT_BACKEND=postgres"
        )
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph-checkpoint-postgres to use PostgreSQL checkpoints."
        ) from exc

    connection_string = settings.supabase_database_url.get_secret_value()
    async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
        await saver.setup()
        yield saver
