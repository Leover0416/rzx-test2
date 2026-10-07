"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It wires the resilience layer (rate limiting, retry with exponential
backoff, circuit breaker) and the observability layer (metrics) into the actual
request flow, and selects the correct per-database executor for each request.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)

# Default timeout (seconds) to wait for a rate-limit slot before failing the request.
_DEFAULT_RATE_LIMIT_TIMEOUT = 30.0


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with exponential backoff, circuit
    breaker pattern, rate limiting, and metric collection.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executor=executor,
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor,
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        *,
        sql_executors: dict[str, SQLExecutor] | None = None,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
        rate_limit_timeout: float = _DEFAULT_RATE_LIMIT_TIMEOUT,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executor: Single SQL execution service (fallback for any database).
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries, backoff and circuit breaker.
            validation_config: Validation configuration including thresholds.
            sql_executors: Optional mapping of database name to its own SQL executor.
                When provided, the executor matching the resolved database is used instead
                of the single fallback ``sql_executor``.
            rate_limiter: Optional multi-rate limiter for queries and LLM calls.
                When provided, rate limits are enforced on the request flow.
            metrics: Optional metrics collector. When provided, request/LLM/security
                metrics are recorded.
            rate_limit_timeout: Seconds to wait for a rate-limit slot before failing.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executor = sql_executor
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.sql_executors = sql_executors or {}
        self.rate_limiter = rate_limiter
        self.metrics = metrics
        self.rate_limit_timeout = rate_limit_timeout

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Validate question length and generate request_id for tracing
        2. Resolve and validate database name
        3. Acquire a query concurrency slot (rate limiting)
        4. Load schema from cache
        5. Generate and validate SQL with retry logic + exponential backoff
        6. Execute SQL against the database's own executor (if return_type == RESULT)
        7. Validate results (optional, under LLM rate limit)
        8. Record metrics and return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Step 0: Enforce configured max question length
        if len(request.question) > self.validation_config.max_question_length:
            error = PgMcpError(
                message=(
                    f"Question exceeds maximum length of "
                    f"{self.validation_config.max_question_length} characters"
                ),
                code=ErrorCode.QUESTION_TOO_LONG,
                details={
                    "max_question_length": self.validation_config.max_question_length,
                    "question_length": len(request.question),
                },
            )
            return self._finalize(self._error_response(error, request_id), "unknown", start_time)

        # Step 1: Resolve database name
        try:
            database_name = self._resolve_database(request.database)
        except PgMcpError as e:
            return self._finalize(self._error_response(e, request_id), "unknown", start_time)

        # Step 2: Acquire a query concurrency slot (rate limiting)
        if self.rate_limiter is not None:
            acquired = await self.rate_limiter.query_limiter.acquire(
                timeout=self.rate_limit_timeout
            )
            if not acquired:
                error = RateLimitExceededError(
                    message="Too many concurrent queries, request rejected by rate limiter",
                    details={"database": database_name},
                )
                return self._finalize(
                    self._error_response(error, request_id), database_name, start_time
                )

        try:
            response = await self._execute(request, request_id, database_name)
            return self._finalize(response, database_name, start_time)
        finally:
            if self.rate_limiter is not None:
                self.rate_limiter.query_limiter.release()

    async def _execute(
        self,
        request: QueryRequest,
        request_id: str,
        database_name: str,
    ) -> QueryResponse:
        """Execute the resolved query pipeline for a known database.

        Args:
            request: Query request.
            request_id: Tracing id.
            database_name: Resolved database name.

        Returns:
            QueryResponse: Structured response.
        """
        logger.info(
            "Starting query execution",
            extra={
                "request_id": request_id,
                "database": database_name,
                "question": request.question[:100],
            },
        )

        try:
            # Step 3: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 4: Generate and validate SQL with retry + backoff
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 5: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 6: Execute SQL against the database's own executor
            logger.debug(
                "Executing SQL",
                extra={"request_id": request_id, "database": database_name},
            )
            executor = self._executor_for(database_name)
            exec_start = time.monotonic()
            results, total_count = await executor.execute(generated_sql)
            execution_time_ms = (time.monotonic() - exec_start) * 1000

            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 7: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 8: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )

        except PgMcpError as e:
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            return self._error_response(e, request_id)
        except Exception as e:
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id, "database": database_name},
            )
            error = PgMcpError(
                message=f"Internal server error: {e!s}",
                code=ErrorCode.INTERNAL_ERROR,
                details={"error_type": type(e).__name__},
            )
            return self._error_response(error, request_id)

    def _executor_for(self, database_name: str) -> SQLExecutor:
        """Select the SQL executor for a given database.

        Prefers the per-database executor when available; falls back to the single
        shared executor.

        Args:
            database_name: Target database name.

        Returns:
            SQLExecutor: Executor configured for the database.

        Raises:
            DatabaseError: If no executor is available for the database.
        """
        if database_name in self.sql_executors:
            return self.sql_executors[database_name]
        if self.sql_executor is not None:
            return self.sql_executor
        raise DatabaseError(
            message=f"No SQL executor available for database '{database_name}'",
            details={"database": database_name},
        )

    def _error_response(self, error: PgMcpError, request_id: str) -> QueryResponse:
        """Build an error QueryResponse from an exception.

        Args:
            error: Caught application error.
            request_id: Tracing id.

        Returns:
            QueryResponse: Error response.
        """
        return QueryResponse(
            success=False,
            generated_sql=None,
            validation=None,
            data=None,
            error=ErrorDetail(
                code=error.code.value,
                message=error.message,
                details=error.details,
            ),
            confidence=0,
            tokens_used=None,
        )

    def _finalize(
        self,
        response: QueryResponse,
        database_name: str,
        start_time: float,
    ) -> QueryResponse:
        """Record metrics for a finished request and return the response.

        Args:
            response: Completed query response.
            database_name: Database that was targeted.
            start_time: Monotonic start time of the request.

        Returns:
            QueryResponse: The unchanged response.
        """
        if self.metrics is not None:
            status = (
                "success"
                if response.success
                else (response.error.code if response.error else "error")
            )
            self.metrics.increment_query_request(status, database_name)
            self.metrics.query_duration.observe(time.monotonic() - start_time)
        return response

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    def _compute_backoff(self, attempt: int) -> float:
        """Compute exponential backoff delay for a retry attempt.

        delay = min(retry_delay * backoff_factor ** attempt, max_backoff)

        Args:
            attempt: Zero-based retry attempt index.

        Returns:
            float: Delay in seconds before the next retry.
        """
        delay = self.resilience_config.retry_delay * (
            self.resilience_config.backoff_factor**attempt
        )
        return min(delay, self.resilience_config.max_backoff)

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (under the LLM rate limiter)
        3. Validates the generated SQL
        4. On validation failure, waits an exponential backoff then retries with feedback
        5. Records success/failure to circuit breaker and metrics

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.
        """
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL under the LLM rate limiter
                llm_start = time.monotonic()
                generated_sql = await self._limited_llm_call(
                    lambda prev=previous_sql, fb=error_feedback: self.sql_generator.generate(
                        question=question,
                        schema=schema,
                        previous_attempt=prev,
                        error_feedback=fb,
                    ),
                    operation="generate_sql",
                    request_id=request_id,
                )
                if self.metrics is not None:
                    self.metrics.observe_llm_latency("generate_sql", time.monotonic() - llm_start)

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if self.metrics is not None:
                        reason = (
                            "security_violation"
                            if isinstance(validation_error, SecurityViolationError)
                            else "sql_parse_error"
                        )
                        self.metrics.increment_sql_rejected(reason)

                    if attempt < max_retries:
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        # Exponential backoff before next attempt
                        await asyncio.sleep(self._compute_backoff(attempt))
                        continue

                    self.circuit_breaker.record_failure()
                    logger.error(
                        "SQL validation failed after all retries",
                        extra={
                            "request_id": request_id,
                            "attempts": attempt + 1,
                            "error": str(validation_error),
                        },
                    )
                    raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build a truthful validation result
                is_write = self.sql_validator.is_write_statement(generated_sql)
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=not is_write,
                    allows_data_modification=is_write,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, SecurityViolationError, SQLParseError):
                raise
            except RateLimitExceededError:
                raise
            except Exception as e:
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _limited_llm_call(
        self,
        call: Any,
        operation: str,
        request_id: str,
    ) -> Any:
        """Run an LLM call under the LLM rate limiter, recording metrics.

        Args:
            call: Zero-arg coroutine factory for the LLM call.
            operation: Metric label for the operation type.
            request_id: Tracing id.

        Returns:
            Any: The result of the LLM call.

        Raises:
            RateLimitExceededError: If the LLM rate limit is exceeded.
        """
        if self.metrics is not None:
            self.metrics.increment_llm_call(operation)

        if self.rate_limiter is not None:
            acquired = await self.rate_limiter.llm_limiter.acquire(
                timeout=self.rate_limit_timeout
            )
            if not acquired:
                raise RateLimitExceededError(
                    message="Too many concurrent LLM calls, request rejected by rate limiter",
                    details={"operation": operation},
                )
            try:
                return await call()
            finally:
                self.rate_limiter.llm_limiter.release()

        return await call()

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM under the LLM rate
        limiter, but failures don't cause the overall query to fail.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            llm_start = time.monotonic()
            validation_result = await self._limited_llm_call(
                lambda: self.result_validator.validate(
                    question=question,
                    sql=sql,
                    results=results,
                    row_count=row_count,
                ),
                operation="validate_result",
                request_id=request_id,
            )
            if self.metrics is not None:
                self.metrics.observe_llm_latency("validate_result", time.monotonic() - llm_start)

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return int(validation_result.confidence)

        except Exception as e:
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000
