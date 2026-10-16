"""Tests for the repository-level compatibility entry point."""

import importlib

from pg_mcp.__main__ import main as package_main


def test_root_main_delegates_to_real_pg_mcp_server() -> None:
    """`python main.py` must not start the placeholder arithmetic server."""
    root_main = importlib.import_module("main")

    assert root_main.main is package_main
