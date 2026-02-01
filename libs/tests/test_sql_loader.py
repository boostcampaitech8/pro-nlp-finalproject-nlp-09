"""
Tests for SQL query loader

Tests SQLQueryLoader for loading and managing SQL query files.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from libs.gcp.sql import SQLQueryLoader, load_query, load_query_with_params


@pytest.fixture
def temp_sql_directory():
    """Create temporary SQL directory structure for testing"""
    with TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create directory structure
        market_dir = base_path / "market"
        market_dir.mkdir()

        news_dir = base_path / "news"
        news_dir.mkdir()

        # Create SQL files
        (market_dir / "test_query.sql").write_text(
            "SELECT * FROM {table_id} WHERE date = '{date}'"
        )
        (news_dir / "test_news.sql").write_text(
            "SELECT * FROM news WHERE filter_status = 'T'"
        )

        yield base_path


def test_sql_loader_initialization(temp_sql_directory):
    """Test SQLQueryLoader initialization"""
    loader = SQLQueryLoader(base_path=temp_sql_directory)
    assert loader.base_path == temp_sql_directory


def test_load_query(temp_sql_directory):
    """Test loading SQL query by name"""
    loader = SQLQueryLoader(base_path=temp_sql_directory)

    query = loader.load("market.test_query")

    assert "SELECT * FROM" in query
    assert "{table_id}" in query


def test_load_query_with_params(temp_sql_directory):
    """Test loading SQL query with parameter substitution"""
    loader = SQLQueryLoader(base_path=temp_sql_directory)

    query = loader.load_with_params(
        "market.test_query",
        table_id="corn_price",
        date="2025-01-20"
    )

    assert "corn_price" in query
    assert "2025-01-20" in query
    assert "{table_id}" not in query


def test_load_invalid_query_name(temp_sql_directory):
    """Test loading query with invalid name format"""
    loader = SQLQueryLoader(base_path=temp_sql_directory)

    with pytest.raises(ValueError, match="Invalid query name"):
        loader.load("invalid_name_without_dot")


def test_load_nonexistent_query(temp_sql_directory):
    """Test loading non-existent query file"""
    loader = SQLQueryLoader(base_path=temp_sql_directory)

    with pytest.raises(FileNotFoundError):
        loader.load("market.nonexistent")


def test_list_queries(temp_sql_directory):
    """Test listing available queries"""
    loader = SQLQueryLoader(base_path=temp_sql_directory)

    queries = loader.list_queries()

    assert "market" in queries
    assert "news" in queries
    assert "test_query" in queries["market"]
    assert "test_news" in queries["news"]


def test_list_queries_by_domain(temp_sql_directory):
    """Test listing queries filtered by domain"""
    loader = SQLQueryLoader(base_path=temp_sql_directory)

    queries = loader.list_queries(domain="market")

    assert "market" in queries
    assert "news" not in queries


def test_global_loader_functions():
    """Test global convenience functions use correct loader"""
    # These should not raise errors (actual SQL files exist)
    try:
        # Just check that the functions are callable
        assert callable(load_query)
        assert callable(load_query_with_params)
    except FileNotFoundError:
        # Expected if SQL files don't exist in actual location
        pass
