"""Tests for the assignment-mandated features.

Covers the three codex-review gaps that this project addresses:
1. Database & security controls now enabled (table/column access limits, EXPLAIN
   policy, per-database executor selection, allow_write_operations).
2. Resilience & observability wired into the request flow (rate limiting,
   exponential backoff, metrics recording, question-length enforcement).
3. Response/model cleanliness (single QueryResponse.to_dict) plus the previously
   unused config fields now taking effect (CacheConfig.max_size eviction).
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import (
    CacheConfig,
    ResilienceConfig,
    SecurityConfig,
    ValidationConfig,
)
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    SQLParseError,
)
from pg_mcp.models.query import QueryRequest, QueryResponse, ReturnType, ValidationResult
from pg_mcp.models.schema import DatabaseSchema
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.sql_validator import SQLValidator

# ---------------------------------------------------------------------------
# 1. SecurityConfig: table/column access limits and EXPLAIN policy are wired
# ---------------------------------------------------------------------------


class TestSecurityConfigAccessControls:
    """SecurityConfig exposes table/column limits and the EXPLAIN policy."""

    def test_defaults(self) -> None:
        """New fields default to permissive-but-safe values."""
        config = SecurityConfig()
        assert config.blocked_tables == []
        assert config.blocked_columns == []
        assert config.allow_explain is False

    def test_custom_values(self) -> None:
        """Explicit lists and EXPLAIN toggle are honoured."""
        config = SecurityConfig(
            blocked_tables=["users", "public.audit_log"],
            blocked_columns=["ssn", "users.email"],
            allow_explain=True,
        )
        assert "users" in config.blocked_tables
        assert "public.audit_log" in config.blocked_tables
        assert config.blocked_columns == ["ssn", "users.email"]
        assert config.allow_explain is True

    def test_list_parsing_from_string(self) -> None:
        """Comma-separated strings are parsed for table/column lists too."""
        config = SecurityConfig(
            blocked_tables="users, orders",
            blocked_columns="ssn, email",
        )
        assert config.blocked_tables == ["users", "orders"]
        assert config.blocked_columns == ["ssn", "email"]


class TestSQLValidatorAccessControls:
    """SQLValidator enforces blocked tables, blocked columns and EXPLAIN policy."""

    @pytest.fixture
    def base_config(self) -> SecurityConfig:
        """Base security config with default values."""
        return SecurityConfig()

    def test_blocked_table_bare(self, base_config: SecurityConfig) -> None:
        """A blocked bare table name is rejected."""
        validator = SQLValidator(base_config, blocked_tables=["secret_table"])
        ok, err = validator.validate("SELECT * FROM secret_table")
        assert not ok
        assert "Access to table 'secret_table'" in err or "not allowed" in err

    def test_blocked_table_schema_qualified(self, base_config: SecurityConfig) -> None:
        """A blocked schema-qualified table is rejected."""
        validator = SQLValidator(base_config, blocked_tables=["public.secret_table"])
        ok, err = validator.validate("SELECT * FROM public.secret_table")
        assert not ok
        assert "not allowed" in err

    def test_blocked_column_bare(self, base_config: SecurityConfig) -> None:
        """A blocked bare column is rejected."""
        validator = SQLValidator(base_config, blocked_columns=["ssn"])
        ok, err = validator.validate("SELECT name, ssn FROM users")
        assert not ok
        assert "ssn" in err or "not allowed" in err

    def test_blocked_column_qualified(self, base_config: SecurityConfig) -> None:
        """A blocked table.column reference is rejected (fully-qualified select)."""
        validator = SQLValidator(base_config, blocked_columns=["users.email"])
        ok, err = validator.validate("SELECT users.email FROM users")
        assert not ok
        assert "email" in err or "not allowed" in err

    def test_allowed_table_and_column_pass(self, base_config: SecurityConfig) -> None:
        """Non-blocked tables/columns are permitted."""
        validator = SQLValidator(base_config, blocked_tables=["secret"], blocked_columns=["ssn"])
        ok, err = validator.validate("SELECT name FROM users")
        assert ok
        assert err is None

    def test_explain_denied_by_default(self, base_config: SecurityConfig) -> None:
        """EXPLAIN is rejected unless explicitly allowed."""
        validator = SQLValidator(base_config)
        ok, err = validator.validate("EXPLAIN SELECT * FROM users")
        assert not ok
        assert "EXPLAIN" in err

    def test_explain_allowed_when_enabled(self, base_config: SecurityConfig) -> None:
        """EXPLAIN passes when allow_explain is enabled."""
        validator = SQLValidator(base_config, allow_explain=True)
        ok, err = validator.validate("EXPLAIN SELECT * FROM users")
        assert ok
        assert err is None

    def test_explain_enabled_via_config(self) -> None:
        """allow_explain falls back to the security config value."""
        config = SecurityConfig(allow_explain=True)
        validator = SQLValidator(config)  # no explicit override
        ok, _err = validator.validate("EXPLAIN SELECT 1")
        assert ok

    def test_dml_denied_when_writes_disabled(self, base_config: SecurityConfig) -> None:
        """INSERT/UPDATE/DELETE are blocked by default."""
        validator = SQLValidator(base_config)
        write_sqls = [
            "INSERT INTO users(name) VALUES ('x')",
            "UPDATE users SET name='x'",
            "DELETE FROM users",
            "MERGE INTO users USING (SELECT 1) s ON true "
            "WHEN MATCHED THEN UPDATE SET name='x'",
        ]
        for sql in write_sqls:
            ok, err = validator.validate(sql)
            assert not ok, sql
            assert "not allowed" in err or "Write operations" in err

    def test_dml_allowed_when_writes_enabled(self) -> None:
        """DML is permitted when SECURITY_ALLOW_WRITE_OPERATIONS is on."""
        config = SecurityConfig(allow_write_operations=True)
        validator = SQLValidator(config)
        ok, err = validator.validate("INSERT INTO users(name) VALUES ('x')")
        assert ok
        assert err is None

    def test_ddl_still_denied_when_writes_enabled(self) -> None:
        """DDL stays forbidden even with write operations enabled."""
        config = SecurityConfig(allow_write_operations=True)
        validator = SQLValidator(config)
        for sql in ("DROP TABLE users", "CREATE TABLE x(id int)", "GRANT SELECT ON users TO bob"):
            ok, err = validator.validate(sql)
            assert not ok, sql
            assert "not allowed" in err

    def test_is_write_statement(self, base_config: SecurityConfig) -> None:
        """is_write_statement classifies read vs write SQL truthfully."""
        validator = SQLValidator(base_config)
        assert validator.is_write_statement("SELECT * FROM users") is False
        assert validator.is_write_statement("INSERT INTO users(name) VALUES ('x')") is True
        assert validator.is_write_statement("UPDATE users SET name='x'") is True
        assert validator.is_write_statement("DELETE FROM users") is True
        # CTE wrapping a write is still a write
        assert validator.is_write_statement(
            "WITH t AS (SELECT 1) UPDATE users SET name='x'"
        ) is True


# ---------------------------------------------------------------------------
# 2. Resilience & observability integrated into the request flow
# ---------------------------------------------------------------------------


class TestOrchestratorExecutorSelection:
    """Per-database executors are selected instead of a single shared one."""

    @pytest.fixture
    def orchestrator(self) -> QueryOrchestrator:
        """Orchestrator with per-db executors plus a fallback."""
        return QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor="fallback_executor",  # single fallback
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db1": MagicMock(), "db2": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            sql_executors={"db1": "executor_db1", "db2": "executor_db2"},
        )

    def test_executor_for_registered_db(self, orchestrator: QueryOrchestrator) -> None:
        """The database's own executor is preferred."""
        assert orchestrator._executor_for("db1") == "executor_db1"
        assert orchestrator._executor_for("db2") == "executor_db2"

    def test_executor_for_unregistered_db_falls_back(self, orchestrator: QueryOrchestrator) -> None:
        """Unregistered databases fall back to the shared executor."""
        assert orchestrator._executor_for("db3") == "fallback_executor"

    def test_executor_for_no_executor_raises(self) -> None:
        """Raises DatabaseError when no executor is available."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=None,
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db1": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )
        with pytest.raises(DatabaseError, match="No SQL executor"):
            orchestrator._executor_for("db1")

    @pytest.mark.asyncio
    async def test_execute_uses_database_executor(self) -> None:
        """execute_query routes execution to the resolved database's executor."""
        db1_executor = AsyncMock()
        db1_executor.execute.return_value = ([{"n": 1}], 1)
        db2_executor = AsyncMock()
        db2_executor.execute.return_value = ([{"n": 2}], 1)

        schema = DatabaseSchema(database_name="db2", tables=[], version="15.0")
        cache = MagicMock()
        cache.get.return_value = schema

        generator = AsyncMock()
        generator.generate.return_value = "SELECT 1;"
        validator = MagicMock()
        validator.validate_or_raise.return_value = None
        validator.is_write_statement.return_value = False

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=db1_executor,  # would be wrong DB if used
            result_validator=AsyncMock(),
            schema_cache=cache,
            pools={"db1": MagicMock(), "db2": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            sql_executors={"db1": db1_executor, "db2": db2_executor},
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="test", database="db2", return_type=ReturnType.RESULT)
        )
        assert response.success is True
        db2_executor.execute.assert_awaited_once_with("SELECT 1;")
        db1_executor.execute.assert_not_called()


class TestOrchestratorBackoff:
    """Exponential backoff is computed from config and applied to retries."""

    def test_compute_backoff_exponential(self) -> None:
        """Delay grows exponentially and is capped by max_backoff."""
        cfg = ResilienceConfig(retry_delay=1.0, backoff_factor=2.0, max_backoff=10.0)
        orch = _minimal_orchestrator(cfg)
        assert orch._compute_backoff(0) == 1.0
        assert orch._compute_backoff(1) == 2.0
        assert orch._compute_backoff(2) == 4.0
        assert orch._compute_backoff(3) == 8.0
        assert orch._compute_backoff(4) == 10.0  # capped

    @pytest.mark.asyncio
    async def test_backoff_applied_between_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Validation failures sleep for the computed delay before retrying."""
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        generator = AsyncMock()
        generator.generate.side_effect = [
            "SELECT * FROM bad;",
            "SELECT * FROM good;",
        ]
        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SQLParseError("parse failed"),
            None,
        ]
        validator.is_write_statement.return_value = False

        orch = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(
                max_retries=3, retry_delay=1.0, backoff_factor=2.0, max_backoff=10.0
            ),
            validation_config=ValidationConfig(),
        )

        sql, _validation, _tokens = await orch._generate_sql_with_retry(
            question="q", schema=MagicMock(), request_id="rid"
        )
        assert sql == "SELECT * FROM good;"
        # One retry happened -> exactly one backoff sleep of 1.0s (attempt=0)
        assert sleeps == [1.0]


class TestOrchestratorRateLimiting:
    """Rate limiting is enforced on the actual request flow."""

    def _orchestrator(
        self, *, query_limit: int = 5, llm_limit: int = 5, **kwargs: object
    ) -> QueryOrchestrator:
        limiter = MultiRateLimiter(query_limit=query_limit, llm_limit=llm_limit)
        return QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executor=AsyncMock(),
            result_validator=AsyncMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            rate_limiter=limiter,
            rate_limit_timeout=0.05,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_query_slot_saturated_rejects(self) -> None:
        """When the query concurrency slot is saturated, the request is rejected."""
        orch = self._orchestrator(query_limit=1, llm_limit=5)
        assert orch.rate_limiter is not None
        # Hold the only query slot.
        held = await orch.rate_limiter.query_limiter.acquire(timeout=0.05)
        assert held is True

        response = await orch.execute_query(
            QueryRequest(question="q", database="db", return_type=ReturnType.RESULT)
        )
        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.RATE_LIMIT_EXCEEDED.value

    @pytest.mark.asyncio
    async def test_llm_slot_saturated_rejects(self) -> None:
        """When the LLM slot is saturated, generation is rejected as rate-limited."""
        orch = self._orchestrator(query_limit=5, llm_limit=1)
        assert orch.rate_limiter is not None
        held = await orch.rate_limiter.llm_limiter.acquire(timeout=0.05)
        assert held is True

        response = await orch.execute_query(
            QueryRequest(question="q", database="db", return_type=ReturnType.RESULT)
        )
        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.RATE_LIMIT_EXCEEDED.value


class TestOrchestratorQuestionLength:
    """Configured max_question_length is enforced (previously unused field)."""

    @pytest.mark.asyncio
    async def test_question_too_long(self) -> None:
        """Oversized questions return a QUESTION_TOO_LONG error."""
        orch = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executor=AsyncMock(),
            result_validator=AsyncMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(max_question_length=5),
        )
        response = await orch.execute_query(
            QueryRequest(question="this is way too long", database="db", return_type=ReturnType.SQL)
        )
        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.QUESTION_TOO_LONG.value


class TestOrchestratorMetrics:
    """Metrics are recorded on the request flow when a collector is provided."""

    @pytest.mark.asyncio
    async def test_success_records_query_metric(self) -> None:
        """A successful query increments the query_requests counter."""
        metrics = MagicMock()
        generator = AsyncMock()
        generator.generate.return_value = "SELECT 1;"
        validator = MagicMock()
        validator.validate_or_raise.return_value = None
        validator.is_write_statement.return_value = False
        cache = MagicMock()
        cache.get.return_value = DatabaseSchema(database_name="db", tables=[], version="15.0")
        executor = AsyncMock()
        executor.execute.return_value = ([{"c": 1}], 1)

        orch = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=executor,
            result_validator=AsyncMock(),
            schema_cache=cache,
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            metrics=metrics,
        )

        response = await orch.execute_query(
            QueryRequest(question="q", database="db", return_type=ReturnType.RESULT)
        )
        assert response.success is True
        metrics.increment_query_request.assert_called_once_with("success", "db")
        metrics.query_duration.observe.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Response/model cleanliness + unused config now active
# ---------------------------------------------------------------------------


class TestQueryResponseToDict:
    """QueryResponse has a single, complete to_dict."""

    def test_success_response_to_dict(self) -> None:
        """to_dict always includes tokens_used and the core response fields."""
        response = QueryResponse(
            success=True,
            generated_sql="SELECT 1;",
            validation=ValidationResult(is_valid=True),
            data=None,
            error=None,
            confidence=90,
            tokens_used=None,
        )
        d = response.to_dict()
        assert d["tokens_used"] == 0  # None coerced to 0
        assert d["success"] is True
        assert d["generated_sql"] == "SELECT 1;"
        assert d["confidence"] == 90
        assert d["error"] is None

    def test_error_response_to_dict(self) -> None:
        """Failed responses carry error and a zero tokens_used."""
        response = QueryResponse(
            success=False,
            generated_sql=None,
            validation=None,
            data=None,
            error={"code": "internal_error", "message": "boom", "details": None},
            confidence=0,
            tokens_used=None,
        )
        d = response.to_dict()
        assert d["tokens_used"] == 0
        assert d["success"] is False
        assert d["error"]["code"] == "internal_error"


class TestSchemaCacheCapacity:
    """CacheConfig.max_size now caps the number of cached schemas (eviction)."""

    def test_evicts_oldest_when_over_capacity(self) -> None:
        """Loading more schemas than max_size evicts the least-recent entries."""
        config = CacheConfig(max_size=2, enabled=True)
        cache = SchemaCache(config)

        # Simulate three cached databases with increasing load timestamps.
        base = datetime.now(UTC)
        for i, name in enumerate(("db_a", "db_b", "db_c")):
            cache._cache[name] = DatabaseSchema(database_name=name, tables=[], version="15.0")
            cache._cache_timestamps[name] = base - timedelta(minutes=10 - i)

        cache._evict_if_over_capacity()

        # db_a is the oldest -> evicted; db_b and db_c remain.
        assert "db_a" not in cache._cache
        assert "db_b" in cache._cache
        assert "db_c" in cache._cache
        assert len(cache._cache) == 2


def _minimal_orchestrator(cfg: ResilienceConfig) -> QueryOrchestrator:
    """Build an orchestrator with only the config we care about."""
    return QueryOrchestrator(
        sql_generator=MagicMock(),
        sql_validator=MagicMock(),
        sql_executor=MagicMock(),
        result_validator=MagicMock(),
        schema_cache=MagicMock(),
        pools={"db": MagicMock()},
        resilience_config=cfg,
        validation_config=ValidationConfig(),
    )
