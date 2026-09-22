"""Common fail-closed Alembic environment helpers for V2 databases."""

from __future__ import annotations

import os
from collections.abc import Callable
from logging.config import fileConfig

import sqlalchemy as sa
from alembic import context
from alembic.config import Config
from sqlalchemy.engine import Connection, make_url

from .specs import DatabaseSpec


def migration_url(config: Config, spec: DatabaseSpec) -> str:
    value = os.environ.get(spec.migration_url_env) or config.get_main_option("sqlalchemy.url")
    if not value or value == "required://set-by-environment":
        raise RuntimeError(f"{spec.migration_url_env} is required for V2 migrations")
    url = make_url(value)
    if url.drivername != "postgresql+psycopg":
        raise RuntimeError(f"{spec.migration_url_env} must use postgresql+psycopg")
    return value


def run_alembic_environment(
    *,
    spec: DatabaseSpec,
    target_metadata: sa.MetaData,
    include_name: Callable[[str | None, str, dict[str, str | None]], bool],
) -> None:
    config = context.config
    if config.config_file_name is not None:
        # Alembic's default ``disable_existing_loggers=True`` silences
        # application loggers in the embedding process.  That made a later
        # pool-governance test (and real migration observability) lose its
        # ERROR signal after an in-process migration.  Keep application
        # loggers alive while still applying the Alembic handler config.
        fileConfig(config.config_file_name, disable_existing_loggers=False)
    url = migration_url(config, spec)

    common_options = {
        "target_metadata": target_metadata,
        "include_schemas": True,
        "include_name": include_name,
        "compare_type": True,
        "compare_server_default": True,
        "version_table": spec.version_table,
        "version_table_schema": spec.control_schema,
        "version_table_pk": True,
        "transaction_per_migration": True,
    }

    if context.is_offline_mode():
        context.configure(
            url=url,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            **common_options,
        )
        with context.begin_transaction():
            context.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {spec.control_schema}"))
            context.run_migrations()
        return

    connectable = sa.create_engine(url, poolclass=sa.pool.NullPool)
    with connectable.connect() as connection:
        _run_online(connection, spec=spec, common_options=common_options)
    connectable.dispose()


def _run_online(connection: Connection, *, spec: DatabaseSpec, common_options: dict[str, object]) -> None:
    context.configure(connection=connection, **common_options)
    with context.begin_transaction():
        context.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {spec.control_schema}"))
        context.run_migrations()
