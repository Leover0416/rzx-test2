"""Regression tests for the second-assignment review feedback."""

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import ValidationError

import pg_mcp.server as server
from pg_mcp.config.settings import (
    DatabaseConfig,
    ObservabilityConfig,
    OpenAIConfig,
    ResilienceConfig,
    Settings,
    ValidationConfig,
)
from pg_mcp.models.errors import LLMError, SQLParseError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.models.schema import DatabaseSchema
from pg_mcp.observability.tracing import get_request_id, request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_generator import SQLGenerationResult


def _database(name: str, port: int) -> DatabaseConfig:
    """Build an isolated database configuration for tests."""
    return DatabaseConfig(
        name=name,
        host=f"{name}.internal",
        port=port,
        user=f"{name}_user",
        password="test-password",
        min_pool_size=1,
        max_pool_size=3,
    )


def _orchestrator(
    generator: Any,
    *,
    circuit_breaker: CircuitBreaker | None = None,
    metrics: Any = None,
) -> QueryOrchestrator:
    """Build an orchestrator with a cached schema and mocked collaborators."""
    cache = MagicMock()
    cache.get.return_value = DatabaseSchema(database_name="primary", tables=[], version="15")
    validator = MagicMock()
    validator.validate_or_raise.return_value = None
    validator.is_write_statement.return_value = False
    return QueryOrchestrator(
        sql_generator=generator,
        sql_validator=validator,
        sql_executor=AsyncMock(),
        result_validator=AsyncMock(),
        schema_cache=cache,
        pools={"primary": MagicMock()},
        resilience_config=ResilienceConfig(max_retries=1, retry_delay=0.1),
        validation_config=ValidationConfig(enabled=False),
        metrics=metrics,
        circuit_breaker=circuit_breaker,
    )


class TestMultiDatabaseSettings:
    """Multi-database configuration is represented and validated in Settings."""

    def test_json_environment_loads_multiple_databases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        databases = [
            {"name": "primary", "host": "db-a", "user": "app", "password": "a"},
            {"name": "analytics", "host": "db-b", "user": "report", "password": "b"},
        ]
        monkeypatch.setenv("DATABASES", json.dumps(databases))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        settings = Settings()

        assert [config.name for config in settings.databases] == ["primary", "analytics"]
        assert settings.database is settings.databases[0]
        assert settings.databases[1].host == "db-b"

    def test_legacy_single_database_is_normalized(self) -> None:
        database = _database("legacy", 5432)
        settings = Settings(
            database=database,
            openai=OpenAIConfig(api_key="sk-test-key"),
        )

        assert settings.databases == [database]
        assert settings.database == database

    def test_duplicate_database_names_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Database names must be unique"):
            Settings(
                databases=[_database("duplicate", 5432), _database("duplicate", 5433)],
                openai=OpenAIConfig(api_key="sk-test-key"),
            )

    def test_explicit_empty_database_list_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="At least one database"):
            Settings(
                databases=[],
                openai=OpenAIConfig(api_key="sk-test-key"),
            )


class TestProductionAssembly:
    """The server lifespan wires every configured database and shared breaker."""

    @pytest.mark.asyncio
    async def test_lifespan_builds_per_database_executors_and_injects_breaker(self) -> None:
        settings = Settings(
            databases=[_database("primary", 5432), _database("analytics", 5433)],
            openai=OpenAIConfig(api_key="sk-test-key"),
            observability=ObservabilityConfig(metrics_enabled=False),
            validation=ValidationConfig(enabled=False),
        )
        pools = {"primary": MagicMock(), "analytics": MagicMock()}
        create_pools = AsyncMock(return_value=pools)
        close_pools = AsyncMock()
        schema_cache = MagicMock()
        schema_cache.load = AsyncMock(
            side_effect=lambda name, _pool: DatabaseSchema(
                database_name=name, tables=[], version="15"
            )
        )
        schema_cache.stop_auto_refresh = AsyncMock()

        def make_executor(**kwargs: object) -> MagicMock:
            executor = MagicMock()
            executor.db_config = kwargs["db_config"]
            return executor

        with (
            patch.object(server, "Settings", return_value=settings),
            patch.object(server, "create_pools", create_pools),
            patch.object(server, "close_pools", close_pools),
            patch.object(server, "SchemaCache", return_value=schema_cache),
            patch.object(server, "SQLGenerator", return_value=MagicMock()),
            patch.object(server, "SQLValidator", return_value=MagicMock()),
            patch.object(server, "SQLExecutor", side_effect=make_executor),
            patch.object(server, "ResultValidator", return_value=MagicMock()),
        ):
            async with server.lifespan(server.mcp):
                assert server._orchestrator is not None
                assert set(server._orchestrator.sql_executors) == {"primary", "analytics"}
                assert server._orchestrator.sql_executors["analytics"].db_config.name == "analytics"
                assert server._orchestrator.circuit_breaker is server._circuit_breaker

        create_pools.assert_awaited_once_with(settings.databases)
        close_pools.assert_awaited_once_with(pools, timeout=5.0)
        assert server._orchestrator is None
        assert server._pools is None

        result = await server.query("must not use closed pools")
        assert result["success"] is False
        assert result["error"]["code"] == "SERVER_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_metrics_server_is_stopped_on_shutdown(self) -> None:
        settings = Settings(
            databases=[_database("primary", 5432)],
            openai=OpenAIConfig(api_key="sk-test-key"),
            observability=ObservabilityConfig(metrics_enabled=True),
            validation=ValidationConfig(enabled=False),
        )
        pool = MagicMock()
        pools = {"primary": pool}
        schema_cache = MagicMock()
        schema_cache.load = AsyncMock(
            return_value=DatabaseSchema(database_name="primary", tables=[], version="15")
        )
        schema_cache.stop_auto_refresh = AsyncMock()
        http_server = MagicMock()
        metrics_thread = MagicMock()
        metrics_thread.is_alive.return_value = False

        with (
            patch.object(server, "Settings", return_value=settings),
            patch.object(server, "create_pools", AsyncMock(return_value=pools)),
            patch.object(server, "close_pools", AsyncMock()),
            patch.object(server, "SchemaCache", return_value=schema_cache),
            patch.object(server, "SQLGenerator", return_value=MagicMock()),
            patch.object(server, "SQLValidator", return_value=MagicMock()),
            patch.object(server, "SQLExecutor", return_value=MagicMock()),
            patch.object(server, "ResultValidator", return_value=MagicMock()),
            patch(
                "prometheus_client.start_http_server",
                return_value=(http_server, metrics_thread),
            ) as start_http_server,
        ):
            async with server.lifespan(server.mcp):
                assert server._metrics_http_server == (http_server, metrics_thread)

        start_http_server.assert_called_once_with(settings.observability.metrics_port)
        http_server.shutdown.assert_called_once_with()
        http_server.server_close.assert_called_once_with()
        metrics_thread.join.assert_called_once_with(timeout=3.0)
        assert server._metrics_http_server is None


class TestTracingAndUsage:
    """Tracing context and token usage propagate through the real request flow."""

    def test_supplied_circuit_breaker_is_used(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
        orchestrator = _orchestrator(AsyncMock(), circuit_breaker=breaker)

        assert orchestrator.circuit_breaker is breaker

    @pytest.mark.asyncio
    async def test_llm_failures_trip_the_supplied_circuit_breaker(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        generator = MagicMock()
        generator.generate = AsyncMock(side_effect=LLMError("provider unavailable"))
        orchestrator = _orchestrator(generator, circuit_breaker=breaker)

        with pytest.raises(LLMError, match="provider unavailable"):
            await orchestrator._generate_sql_with_retry(
                question="query",
                schema=MagicMock(),
                request_id="request-id",
            )

        assert breaker.state.value == "open"
        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_request_context_is_active_and_cleaned_up(self) -> None:
        seen_request_ids: list[str | None] = []

        async def generate(**_kwargs: object) -> SQLGenerationResult:
            seen_request_ids.append(get_request_id())
            return SQLGenerationResult("SELECT 1;", 23)

        generator = MagicMock()
        generator.generate = AsyncMock(side_effect=generate)
        orchestrator = _orchestrator(generator)

        response = await orchestrator.execute_query(
            QueryRequest(question="one", database="primary", return_type=ReturnType.SQL)
        )

        assert response.success is True
        assert response.tokens_used == 23
        request_id = seen_request_ids[0]
        assert request_id is not None
        uuid.UUID(request_id)
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_existing_request_context_is_preserved(self) -> None:
        seen_request_ids: list[str | None] = []

        async def generate(**_kwargs: object) -> SQLGenerationResult:
            seen_request_ids.append(get_request_id())
            return SQLGenerationResult("SELECT 1;", 1)

        generator = MagicMock()
        generator.generate = AsyncMock(side_effect=generate)
        orchestrator = _orchestrator(generator)

        async with request_context("upstream-request"):
            await orchestrator.execute_query(
                QueryRequest(question="one", database="primary", return_type=ReturnType.SQL)
            )
            assert get_request_id() == "upstream-request"

        assert seen_request_ids == ["upstream-request"]
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_retry_aggregates_and_records_token_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        generator = AsyncMock()
        generator.generate.side_effect = [
            SQLGenerationResult("SELECT bad;", 11),
            SQLGenerationResult("SELECT good;", 17),
        ]
        metrics = MagicMock()
        orchestrator = _orchestrator(generator, metrics=metrics)
        orchestrator.sql_validator.validate_or_raise.side_effect = [
            SQLParseError("bad SQL"),
            None,
        ]
        sleep = AsyncMock()
        monkeypatch.setattr("pg_mcp.services.orchestrator.asyncio.sleep", sleep)

        sql, _validation, tokens_used = await orchestrator._generate_sql_with_retry(
            question="query", schema=MagicMock(), request_id="request-id"
        )

        assert sql == "SELECT good;"
        assert tokens_used == 28
        assert metrics.increment_llm_tokens.call_args_list == [
            call("generate_sql", 11),
            call("generate_sql", 17),
        ]


def test_query_request_does_not_override_configurable_length_limit() -> None:
    """The model ceiling must not make ValidationConfig values above 10k unusable."""
    request = QueryRequest(question="x" * 10001, return_type=ReturnType.SQL)

    assert len(request.question) == 10001
