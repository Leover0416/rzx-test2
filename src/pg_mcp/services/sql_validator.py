"""SQL Security Validator using SQLGlot.

This module provides SQL validation and security checking using SQLGlot parser.
It ensures that only safe, read-only queries are executed and blocks potentially
dangerous operations.
"""

import re
from typing import ClassVar

import sqlglot
from sqlglot import exp

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.models.errors import SecurityViolationError, SQLParseError


class SQLValidator:
    """SQL security validator using SQLGlot for parsing and validation.

    This validator ensures queries are safe by:
    - Allowing only SELECT statements
    - Blocking dangerous functions (pg_sleep, file operations, etc.)
    - Preventing access to blocked tables and columns
    - Rejecting multi-statement queries
    - Validating subquery safety
    """

    # Allowed statement types at the top level (including set operations)
    ALLOWED_STATEMENT_TYPES: ClassVar = {exp.Select, exp.Union, exp.Intersect, exp.Except}

    # Write (DML) statement types - only allowed when allow_write_operations is enabled
    WRITE_STATEMENT_TYPES: ClassVar = {
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Merge,
    }

    # Statements never allowed regardless of configuration (DDL/admin).
    # Note: TRUNCATE is parsed as a Command and rejected by the command check.
    ALWAYS_FORBIDDEN_STATEMENT_TYPES: ClassVar = {
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Grant,
        exp.Revoke,
        exp.Set,
        exp.Command,
        exp.Use,
    }

    # Built-in dangerous PostgreSQL functions
    BUILTIN_DANGEROUS_FUNCTIONS: ClassVar = {
        "pg_sleep",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "dblink_open",
        "pg_write_file",
        "pg_execute_sql",
        "copy_from",
        "copy_to",
    }

    def __init__(
        self,
        config: SecurityConfig,
        blocked_tables: list[str] | None = None,
        blocked_columns: list[str] | None = None,
        allow_explain: bool | None = None,
    ) -> None:
        """Initialize SQL validator.

        Args:
            config: Security configuration containing blocked functions and settings.
            blocked_tables: Optional list of table names to block access to.
                Defaults to ``config.blocked_tables`` when not provided.
            blocked_columns: Optional list of column names to block access to.
                Defaults to ``config.blocked_columns`` when not provided.
            allow_explain: Whether to allow EXPLAIN statements.
                Defaults to ``config.allow_explain`` when not provided.
        """
        self.config = config
        self.blocked_tables = {
            t.lower()
            for t in (blocked_tables if blocked_tables is not None else config.blocked_tables)
        }
        self.blocked_columns = {
            c.lower()
            for c in (blocked_columns if blocked_columns is not None else config.blocked_columns)
        }
        self.allow_explain = allow_explain if allow_explain is not None else config.allow_explain

        # Combine built-in dangerous functions with custom blocked functions
        self.blocked_functions = self.BUILTIN_DANGEROUS_FUNCTIONS | {
            f.lower() for f in config.blocked_functions
        }

    def validate(self, sql: str) -> tuple[bool, str | None]:
        """Validate SQL query for security compliance.

        Args:
            sql: SQL query string to validate.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        try:
            self.validate_or_raise(sql)
            return (True, None)
        except (SecurityViolationError, SQLParseError) as e:
            return (False, str(e))

    def validate_or_raise(self, sql: str) -> None:
        """Validate SQL query and raise exception on violation.

        Args:
            sql: SQL query string to validate.

        Raises:
            SQLParseError: If SQL cannot be parsed.
            SecurityViolationError: If SQL violates security constraints.
        """
        # Check for empty or whitespace-only SQL
        if not sql or not sql.strip():
            raise SQLParseError("SQL query cannot be empty")

        # Parse SQL using SQLGlot
        try:
            parsed = sqlglot.parse(sql, read="postgres")
        except Exception as e:
            raise SQLParseError(f"Failed to parse SQL: {e}") from e

        # Check for multiple statements
        if len(parsed) > 1:
            raise SecurityViolationError(
                "Multiple statements not allowed. Only single SELECT queries are permitted."
            )

        if not parsed:
            raise SQLParseError("No valid SQL statement found")

        statement = parsed[0]

        # Check for null or empty statement (e.g., comment-only SQL)
        if statement is None or isinstance(statement, type(None)):
            raise SQLParseError("No valid SQL statement found")

        # Handle EXPLAIN statements (parsed as Command in sqlglot 28.5.0)
        if isinstance(statement, exp.Command):
            # Check if it's an EXPLAIN command
            cmd_name = str(statement.this).upper() if statement.this else ""
            if cmd_name == "EXPLAIN":
                if not self.allow_explain:
                    raise SecurityViolationError("EXPLAIN statements are not allowed")
                inner_sql = self._extract_explain_query(statement)
                # Validate the explained statement with the same table, column,
                # function, statement-type, and multi-statement policy.
                self.validate_or_raise(inner_sql)
                return
            else:
                # Other commands are not allowed
                raise SecurityViolationError(
                    f"Command '{cmd_name}' is not allowed. Only SELECT queries are permitted."
                )

        # Handle CTE (WITH) statements - extract the main query
        if isinstance(statement, exp.With):
            # WITH statements are allowed, but we need to validate the main query
            if statement.this:
                main_query = statement.this
            else:
                raise SQLParseError("WITH statement has no main query")
        else:
            main_query = statement

        # Perform security checks
        if error := self._check_statement_type(main_query):
            raise SecurityViolationError(error)

        if error := self._check_dangerous_functions(statement):
            raise SecurityViolationError(error)

        if error := self._check_blocked_tables(statement):
            raise SecurityViolationError(error)

        if error := self._check_blocked_columns(statement):
            raise SecurityViolationError(error)

        if error := self._check_subquery_safety(statement):
            raise SecurityViolationError(error)

    def _check_statement_type(self, statement: exp.Expression) -> str | None:
        """Check if statement type is allowed.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        # Never allow DDL/admin statements regardless of write policy
        for forbidden_type in self.ALWAYS_FORBIDDEN_STATEMENT_TYPES:
            if isinstance(statement, forbidden_type):
                stmt_name = forbidden_type.__name__.upper()
                return f"{stmt_name} statements are not allowed. Only SELECT queries are permitted."

        # Determine allowed set based on write policy
        allowed_types: set[type[exp.Expression]] = set(self.ALLOWED_STATEMENT_TYPES)
        if self.config.allow_write_operations:
            allowed_types |= self.WRITE_STATEMENT_TYPES

        if isinstance(statement, tuple(allowed_types)):
            return None

        # Write statements are gated by the write policy
        if isinstance(statement, tuple(self.WRITE_STATEMENT_TYPES)):
            return (
                "Write operations (INSERT/UPDATE/DELETE/MERGE) are not allowed. "
                "Enable SECURITY_ALLOW_WRITE_OPERATIONS to permit them."
            )

        stmt_type = type(statement).__name__
        return f"Statement type {stmt_type} is not allowed. Only SELECT queries are permitted."

    def _check_dangerous_functions(self, statement: exp.Expression) -> str | None:
        """Check for use of blocked/dangerous functions.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        # Find all function calls in the query
        for func in statement.find_all(exp.Func):
            func_name = func.name.lower() if func.name else ""

            if func_name in self.blocked_functions:
                return f"Function '{func_name}' is blocked for security reasons"

        return None

    def _check_blocked_tables(self, statement: exp.Expression) -> str | None:
        """Check for access to blocked tables.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        if not self.blocked_tables:
            return None

        # Find all table references
        for table in statement.find_all(exp.Table):
            table_name = table.name.lower() if table.name else ""

            # Check for exact match on bare table name
            if table_name in self.blocked_tables:
                return f"Access to table '{table_name}' is not allowed"

            # Check for schema-qualified name (schema.table)
            if table.db:
                qualified_name = f"{table.db.lower()}.{table_name}"
                if qualified_name in self.blocked_tables:
                    return f"Access to table '{qualified_name}' is not allowed"

        return None

    def _check_blocked_columns(self, statement: exp.Expression) -> str | None:
        """Check for access to blocked columns.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        if not self.blocked_columns:
            return None

        table_aliases: dict[str, str] = {}
        referenced_tables: set[str] = set()
        for table in statement.find_all(exp.Table):
            table_name = table.name.lower() if table.name else ""
            if not table_name:
                continue
            referenced_tables.add(table_name)
            table_aliases[table_name] = table_name
            if table.alias:
                table_aliases[table.alias.lower()] = table_name

        # A projection wildcard can expand to a blocked column without naming
        # it in the AST. Reject SELECT * / table.* conservatively whenever a
        # column deny-list is configured; COUNT(*) remains safe and allowed.
        for select in statement.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star) or (
                    isinstance(projection, exp.Column) and projection.is_star
                ):
                    return (
                        "Wildcard column selection is not allowed when blocked columns "
                        "are configured"
                    )

        # Find all column references
        for column in statement.find_all(exp.Column):
            column_name = column.name.lower() if column.name else ""

            # Check for exact match
            if column_name in self.blocked_columns:
                return f"Access to column '{column_name}' is not allowed"

            # Check for qualified column names (table.column)
            if column.table:
                qualifier = column.table.lower()
                qualified_name = f"{qualifier}.{column_name}"
                physical_table = table_aliases.get(qualifier, qualifier)
                physical_name = f"{physical_table}.{column_name}"
                if qualified_name in self.blocked_columns or physical_name in self.blocked_columns:
                    return f"Access to column '{physical_name}' is not allowed"
            else:
                # A bare column may still resolve to a table-qualified deny-list
                # entry. Match it against every physical table referenced by the
                # statement rather than allowing aliases/unqualified syntax to bypass.
                for table_name in referenced_tables:
                    qualified_name = f"{table_name}.{column_name}"
                    if qualified_name in self.blocked_columns:
                        return f"Access to column '{qualified_name}' is not allowed"

        return None

    def _extract_explain_query(self, statement: exp.Command) -> str:
        """Extract and safely constrain the statement wrapped by EXPLAIN.

        SQLGlot currently represents PostgreSQL EXPLAIN as a generic Command.
        ``EXPLAIN ANALYZE`` is never accepted because PostgreSQL executes the
        wrapped statement when ANALYZE is enabled.
        """
        expression = statement.expression
        payload = str(expression.this) if isinstance(expression, exp.Literal) else ""
        payload = payload.strip()
        if not payload:
            raise SQLParseError("EXPLAIN statement has no query")

        parenthesized = re.match(r"^\((?P<options>[^)]*)\)\s*(?P<query>.+)$", payload, re.DOTALL)
        if parenthesized is not None:
            options = parenthesized.group("options")
            if re.search(r"\bANALYZE\b", options, re.IGNORECASE):
                raise SecurityViolationError(
                    "EXPLAIN ANALYZE is not allowed because it executes the query"
                )
            return parenthesized.group("query").strip()

        # PostgreSQL also accepts the legacy `EXPLAIN [ANALYZE] [VERBOSE]`
        # form. Strip VERBOSE, but fail closed on ANALYZE.
        remaining = payload
        while True:
            option = re.match(r"^(?P<option>ANALYZE|VERBOSE)\b\s*(?P<rest>.*)$", remaining, re.I)
            if option is None:
                break
            if option.group("option").upper() == "ANALYZE":
                raise SecurityViolationError(
                    "EXPLAIN ANALYZE is not allowed because it executes the query"
                )
            remaining = option.group("rest").strip()

        if not remaining:
            raise SQLParseError("EXPLAIN statement has no query")
        return remaining

    def _check_subquery_safety(self, statement: exp.Expression) -> str | None:
        """Check that all CTEs and subqueries obey the statement policy.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        # Data-modifying CTEs are legal PostgreSQL and can hide writes behind
        # a top-level SELECT, so validate every CTE body explicitly.
        for cte in statement.find_all(exp.CTE):
            if cte.this is None:
                return "CTE has no query body"
            if error := self._check_statement_type(cte.this):
                return f"Unsafe CTE: {error}"

        # Find all subqueries
        for subquery in statement.find_all(exp.Subquery):
            if subquery.this:
                inner_stmt = subquery.this

                # Check if the inner statement is a never-allowed type
                for forbidden_type in self.ALWAYS_FORBIDDEN_STATEMENT_TYPES:
                    if isinstance(inner_stmt, forbidden_type):
                        stmt_name = forbidden_type.__name__.upper()
                        return f"{stmt_name} statements in subqueries are not allowed"

                # Subqueries must resolve to SELECT/WITH (DML in a subquery is invalid SQL)
                if not isinstance(inner_stmt, (exp.Select, exp.With)):
                    return "Subqueries must contain only SELECT statements"

        return None

    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL query to a canonical form.

        This removes extra whitespace, standardizes formatting, and makes
        queries easier to compare or cache.

        Args:
            sql: SQL query string to normalize.

        Returns:
            Normalized SQL string.

        Raises:
            SQLParseError: If SQL cannot be parsed.
        """
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
            # Generate normalized SQL
            return parsed.sql(dialect="postgres", pretty=False)
        except Exception as e:
            raise SQLParseError(f"Failed to normalize SQL: {e}") from e

    def is_write_statement(self, sql: str) -> bool:
        """Determine whether a SQL statement performs a data-modifying operation.

        Args:
            sql: SQL query string.

        Returns:
            bool: True if the statement is INSERT/UPDATE/DELETE/MERGE.

        Raises:
            SQLParseError: If SQL cannot be parsed.
        """
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
        except Exception as e:
            raise SQLParseError(f"Failed to parse SQL: {e}") from e

        statement: exp.Expression = parsed
        return any(isinstance(node, tuple(self.WRITE_STATEMENT_TYPES)) for node in statement.walk())

    def extract_tables(self, sql: str) -> list[str]:
        """Extract all table names referenced in the SQL query.

        Args:
            sql: SQL query string.

        Returns:
            List of table names (in lowercase).

        Raises:
            SQLParseError: If SQL cannot be parsed.
        """
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
            tables = []

            for table in parsed.find_all(exp.Table):
                if table.name:
                    tables.append(table.name.lower())

            return sorted(set(tables))
        except Exception as e:
            raise SQLParseError(f"Failed to extract tables: {e}") from e
