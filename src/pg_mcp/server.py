"""FastMCP server for PostgreSQL natural language query interface.

This module implements the MCP server using FastMCP, exposing the query
functionality as an MCP tool. It includes complete lifespan management for
initializing and cleaning up all components.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from asyncpg import Pool
from mcp.server.fastmcp import FastMCP

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import Settings
from pg_mcp.db.pool import close_pools, create_pools
from pg_mcp.models.query import QueryRequest, QueryResponse, ReturnType
from pg_mcp.observability.logging import configure_logging, get_logger
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = get_logger(__name__)

# Global state for lifespan management
_settings: Settings | None = None
_pools: dict[str, Pool] | None = None
_schema_cache: SchemaCache | None = None
_orchestrator: QueryOrchestrator | None = None
_metrics: MetricsCollector | None = None
_circuit_breaker: CircuitBreaker | None = None
_rate_limiter: MultiRateLimiter | None = None
_metrics_http_server: tuple[Any, Any] | None = None


@asynccontextmanager
async def lifespan(_app: FastMCP) -> AsyncIterator[None]:
    """Lifespan context manager for server initialization and cleanup.

    This function manages the complete lifecycle of the MCP server:

    Startup:
        1. Load configuration from Settings
        2. Configure logging
        3. Create database connection pools
        4. Load schema cache for all databases
        5. Initialize metrics collector
        6. Create service components (generators, validators, executors)
        7. Initialize resilience components (circuit breaker, rate limiter)
        8. Create query orchestrator
        9. Start metrics HTTP server (optional)

    Shutdown:
        1. Stop schema auto-refresh (if enabled)
        2. Close all database connection pools
        3. Stop metrics HTTP server (if running)

    Yields:
        None

    Example:
        >>> async with lifespan():
        ...     # Server is running with all components initialized
        ...     pass
    """
    global _settings, _pools, _schema_cache, _orchestrator, _metrics
    global _circuit_breaker, _rate_limiter, _metrics_http_server

    logger.info("Starting PostgreSQL MCP Server initialization...")

    # Never expose components left over from an earlier or failed lifespan.
    _settings = None
    _pools = None
    _schema_cache = None
    _orchestrator = None
    _metrics = None
    _circuit_breaker = None
    _rate_limiter = None
    _metrics_http_server = None

    try:
        # 1. Load Settings
        logger.info("Loading configuration...")
        _settings = Settings()

        # 2. Configure logging
        logger.info("Configuring logging...")
        configure_logging(
            level=_settings.observability.log_level,
            log_format=_settings.observability.log_format,
            enable_sensitive_filter=True,
        )

        logger.info(
            "Configuration loaded",
            extra={
                "environment": _settings.environment,
                "log_level": _settings.observability.log_level,
            },
        )

        # 3. Create database connection pools
        logger.info("Creating database connection pools...")
        database_configs = {config.name: config for config in _settings.databases}
        _pools = await create_pools(_settings.databases)
        for db_name, config in database_configs.items():
            logger.info(
                f"Created connection pool for database '{db_name}'",
                extra={
                    "min_size": config.min_pool_size,
                    "max_size": config.max_pool_size,
                },
            )

        # 4. Load Schema cache
        logger.info("Initializing schema cache...")
        _schema_cache = SchemaCache(_settings.cache)

        for db_name, pool in _pools.items():
            logger.info(f"Loading schema for database '{db_name}'...")
            schema = await _schema_cache.load(db_name, pool)
            logger.info(
                f"Schema loaded for '{db_name}'",
                extra={
                    "tables": len(schema.tables),
                },
            )

        # Optional: Start schema auto-refresh
        # Disabled by default to avoid unnecessary background tasks
        # Uncomment to enable:
        # if _settings.cache.enabled:
        #     logger.info("Starting schema auto-refresh...")
        #     await _schema_cache.start_auto_refresh(
        #         interval_minutes=60,  # Refresh every hour
        #         pools=_pools,
        #     )

        # 5. Initialize metrics collector
        logger.info("Initializing metrics collector...")
        _metrics = MetricsCollector()

        # Start metrics HTTP server if enabled
        if _settings.observability.metrics_enabled:
            from prometheus_client import start_http_server

            _metrics_http_server = start_http_server(_settings.observability.metrics_port)
            logger.info(f"Metrics server started on port {_settings.observability.metrics_port}")

        # 6. Create service components
        logger.info("Initializing service components...")

        # SQL Generator
        sql_generator = SQLGenerator(_settings.openai)

        # SQL Validator (table/column access limits and EXPLAIN policy from settings)
        sql_validator = SQLValidator(
            config=_settings.security,
            blocked_tables=_settings.security.blocked_tables,
            blocked_columns=_settings.security.blocked_columns,
            allow_explain=_settings.security.allow_explain,
        )

        # SQL Executor (create one per database)
        sql_executors: dict[str, SQLExecutor] = {}
        for db_name, pool in _pools.items():
            db_config = database_configs[db_name]
            executor = SQLExecutor(
                pool=pool,
                security_config=_settings.security,
                db_config=db_config,
                metrics=_metrics,
            )
            sql_executors[db_name] = executor
            # Report connection pool headroom as a gauge for monitoring
            _metrics.set_db_connections_active(db_name, db_config.max_pool_size)
            logger.info(f"Created SQL executor for database '{db_name}'")

        # Result Validator
        result_validator = ResultValidator(
            openai_config=_settings.openai,
            validation_config=_settings.validation,
        )

        # 7. Initialize resilience components
        logger.info("Initializing resilience components...")

        # Circuit Breaker for LLM calls
        _circuit_breaker = CircuitBreaker(
            failure_threshold=_settings.resilience.circuit_breaker_threshold,
            recovery_timeout=_settings.resilience.circuit_breaker_timeout,
        )

        # Rate Limiter (query and LLM concurrency from settings)
        _rate_limiter = MultiRateLimiter(
            query_limit=_settings.resilience.query_concurrency,
            llm_limit=_settings.resilience.llm_concurrency,
        )

        # 8. Create QueryOrchestrator
        logger.info("Creating query orchestrator...")
        primary_database = _settings.databases[0].name
        _orchestrator = QueryOrchestrator(
            sql_generator=sql_generator,
            sql_validator=sql_validator,
            sql_executor=sql_executors[primary_database],  # Backward-compatible fallback
            result_validator=result_validator,
            schema_cache=_schema_cache,
            pools=_pools,
            resilience_config=_settings.resilience,
            validation_config=_settings.validation,
            sql_executors=sql_executors,  # Per-database executors
            rate_limiter=_rate_limiter,  # Enforce query/LLM concurrency limits
            metrics=_metrics,  # Record request/LLM/security metrics
            circuit_breaker=_circuit_breaker,  # Use the lifecycle-managed breaker
        )

        logger.info("PostgreSQL MCP Server initialization complete!")
        logger.info(
            "Server ready to accept requests",
            extra={
                "databases": list(_pools.keys()),
                "cache_enabled": _settings.cache.enabled,
                "metrics_enabled": _settings.observability.metrics_enabled,
            },
        )

        # Yield to run the server
        yield

    finally:
        # Shutdown sequence
        logger.info("Starting PostgreSQL MCP Server shutdown...")

        # Stop schema auto-refresh with timeout
        if _schema_cache is not None:
            try:
                import asyncio

                await asyncio.wait_for(_schema_cache.stop_auto_refresh(), timeout=3.0)
                logger.info("Schema auto-refresh stopped")
            except TimeoutError:
                logger.warning("Schema auto-refresh stop timed out")
            except Exception as e:
                logger.warning(f"Error stopping schema auto-refresh: {e!s}")

        # Close database connection pools with timeout
        if _pools is not None:
            try:
                # Use 5 second timeout for graceful shutdown
                await close_pools(_pools, timeout=5.0)
                logger.info("Database connection pools closed")
            except Exception as e:
                logger.error(f"Error closing connection pools: {e!s}")

        # prometheus_client starts a background WSGI server and returns both
        # the server and thread. Stop both so sequential lifespans can reuse
        # the configured port without leaking a daemon thread.
        if _metrics_http_server is not None:
            http_server, metrics_thread = _metrics_http_server
            try:
                http_server.shutdown()
                http_server.server_close()
                metrics_thread.join(timeout=3.0)
                if metrics_thread.is_alive():
                    logger.warning("Metrics HTTP server thread did not stop within timeout")
                else:
                    logger.info("Metrics HTTP server stopped")
            except Exception as e:
                logger.warning(f"Error stopping metrics HTTP server: {e!s}")

        # Components above own resources that are now closed. Clear every
        # global so calls after shutdown fail closed instead of using stale
        # executors, pools, or resilience state.
        _orchestrator = None
        _schema_cache = None
        _pools = None
        _settings = None
        _metrics = None
        _circuit_breaker = None
        _rate_limiter = None
        _metrics_http_server = None

        logger.info("PostgreSQL MCP Server shutdown complete")


# Create FastMCP server instance with lifespan
mcp = FastMCP("pg-mcp", lifespan=lifespan)


@mcp.tool()
async def query(
    question: str,
    database: str | None = None,
    return_type: str = "result",
) -> dict[str, Any]:
    """Execute a natural language query against PostgreSQL database.

    This tool converts natural language questions into SQL queries and executes
    them against the specified PostgreSQL database. It includes comprehensive
    security validation, result verification, and error handling.

    Args:
        question: Natural language description of the query.
            Examples:
                - "How many users registered in the last 30 days?"
                - "Show me the top 10 products by revenue"
                - "What is the average order value by country?"

        database: Target database name (optional if only one database is configured).
            If not specified and only one database is available, it will be
            automatically selected.

        return_type: Type of result to return.
            Options:
                - "sql": Return only the generated SQL query without executing it
                - "result": Execute the query and return results (default)

    Returns:
        dict: Query response containing:
            - success (bool): Whether the query succeeded
            - generated_sql (str): The generated SQL query
            - data (dict): Query results if executed (columns, rows, row_count, etc.)
            - error (dict): Error information if query failed
            - confidence (int): Confidence score (0-100) for result quality
            - tokens_used (int): Number of LLM tokens consumed

    Examples:
        >>> # Get query results
        >>> result = await query(
        ...     question="How many active users are there?",
        ...     return_type="result"
        ... )
        >>> print(result["data"]["rows"])

        >>> # Get SQL only
        >>> result = await query(
        ...     question="Count all products",
        ...     return_type="sql"
        ... )
        >>> print(result["generated_sql"])

    Raises:
        This function does not raise exceptions. All errors are captured and
        returned in the response with success=False and error details.

    Security:
        - Only SELECT queries are allowed (DML gated by SECURITY_ALLOW_WRITE_OPERATIONS)
        - Dangerous PostgreSQL functions are blocked (pg_sleep, file operations, etc.)
        - Access to blocked tables/columns is rejected (SECURITY_BLOCKED_TABLES/COLUMNS)
        - EXPLAIN is denied unless SECURITY_ALLOW_EXPLAIN is enabled
        - Query execution timeout is enforced
        - Row count limits prevent memory exhaustion
        - Transactions are read-only unless SECURITY_ALLOW_WRITE_OPERATIONS is enabled
    """
    global _orchestrator

    if _orchestrator is None:
        return {
            "success": False,
            "error": {
                "code": "SERVER_NOT_INITIALIZED",
                "message": "Server not initialized properly",
                "details": None,
            },
        }

    # Validate return_type
    if return_type not in ("sql", "result"):
        return {
            "success": False,
            "error": {
                "code": "INVALID_PARAMETER",
                "message": f"Invalid return_type: '{return_type}'. Must be 'sql' or 'result'.",
                "details": {"return_type": return_type},
            },
        }

    # Build request
    try:
        request = QueryRequest(
            question=question,
            database=database,
            return_type=ReturnType(return_type),
        )
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": f"Invalid request parameters: {e!s}",
                "details": {"error": str(e)},
            },
        }

    # Execute query through orchestrator
    try:
        response: QueryResponse = await _orchestrator.execute_query(request)
        result = response.to_dict()
        # Ensure tokens_used is always present
        if "tokens_used" not in result:
            result["tokens_used"] = 0
        return result
    except Exception as e:
        logger.exception("Unexpected error in query tool")
        return {
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"Internal server error: {e!s}",
                "details": {"error_type": type(e).__name__},
            },
            "tokens_used": 0,
        }


if __name__ == "__main__":
    """Run the server when executed directly."""
    import anyio

    anyio.run(mcp.run_stdio_async)
