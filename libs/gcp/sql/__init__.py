"""
SQL query management utilities

This module provides utilities for loading and managing SQL query files.
Queries are organized by domain (market, news) for better maintainability.
"""

from pathlib import Path
from typing import Dict, Any


class SQLQueryLoader:
    """
    SQL query file loader with parameter substitution

    This class provides a clean interface for loading SQL queries from files
    and substituting parameters.
    """

    def __init__(self, base_path: Path = None):
        """
        Initialize SQL query loader

        Args:
            base_path: Base path for SQL files (default: libs/gcp/sql/)
        """
        if base_path is None:
            # Default to the directory containing this file
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)

    def load(self, query_name: str) -> str:
        """
        Load SQL query by name

        Args:
            query_name: Query name in format "domain.query_file"
                       (e.g., "market.get_prophet_features")

        Returns:
            str: SQL query content

        Example:
            >>> loader = SQLQueryLoader()
            >>> query = loader.load("market.get_prophet_features")
        """
        parts = query_name.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid query name: {query_name}. Expected format: 'domain.query_file'"
            )

        domain, query_file = parts
        file_path = self.base_path / domain / f"{query_file}.sql"

        if not file_path.exists():
            raise FileNotFoundError(
                f"SQL file not found: {file_path}\nQuery name: {query_name}"
            )

        return file_path.read_text()

    def load_with_params(self, query_name: str, **params) -> str:
        """
        Load SQL query and substitute parameters

        Args:
            query_name: Query name in format "domain.query_file"
            **params: Parameters for string formatting

        Returns:
            str: SQL query with parameters substituted

        Example:
            >>> loader = SQLQueryLoader()
            >>> query = loader.load_with_params(
            ...     "market.get_prophet_features",
            ...     project_id="my-project",
            ...     dataset_id="corn",
            ...     table_id="corn_price",
            ...     date_column="time",
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-20"
            ... )
        """
        query = self.load(query_name)
        if params:
            query = query.format(**params)
        return query

    def list_queries(self, domain: str = None) -> Dict[str, list]:
        """
        List available SQL query files

        Args:
            domain: Filter by domain (if None, lists all)

        Returns:
            Dict[str, list]: Dictionary mapping domain to query file names

        Example:
            >>> loader = SQLQueryLoader()
            >>> queries = loader.list_queries()
            >>> print(queries)
            {'market': ['get_prophet_features', 'get_price_history'], ...}
        """
        if domain:
            domains = [domain]
        else:
            # List all subdirectories
            domains = [d.name for d in self.base_path.iterdir() if d.is_dir()]

        result = {}
        for dom in domains:
            domain_path = self.base_path / dom
            if domain_path.exists():
                sql_files = [f.stem for f in domain_path.glob("*.sql")]
                result[dom] = sorted(sql_files)

        return result


# Global loader instance for convenience
sql_loader = SQLQueryLoader()


def load_query(query_name: str) -> str:
    """
    Convenience function to load SQL query

    Args:
        query_name: Query name in format "domain.query_file"

    Returns:
        str: SQL query content

    Example:
        >>> from libs.gcp.sql import load_query
        >>> query = load_query("market.get_prophet_features")
    """
    return sql_loader.load(query_name)


def load_query_with_params(query_name: str, **params) -> str:
    """
    Convenience function to load SQL query with parameters

    Args:
        query_name: Query name in format "domain.query_file"
        **params: Parameters for string formatting

    Returns:
        str: SQL query with parameters substituted

    Example:
        >>> from libs.gcp.sql import load_query_with_params
        >>> query = load_query_with_params(
        ...     "market.get_prophet_features",
        ...     table_id="corn_price",
        ...     start_date="2025-01-01"
        ... )
    """
    return sql_loader.load_with_params(query_name, **params)
