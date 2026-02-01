"""
Integration tests for BigQuery daily_prices table connectivity

These tests require:
1. Valid GCP credentials (gcloud auth application-default login)
2. Access to BigQuery dataset with daily_prices table
3. .env file with GCP_PROJECT_ID and BIGQUERY_DATASET_ID

Run with: pytest tests/test_connection.py -v -s
Use -s flag to see print output
Use --skip-integration to skip these tests
"""

import pytest
from libs.utils.config import get_config
from libs.gcp.bigquery import BigQueryService


pytestmark = pytest.mark.integration  # Mark all tests as integration tests


@pytest.fixture(scope="module")
def config():
    """Load configuration for integration tests"""
    return get_config()


@pytest.fixture(scope="module")
def bq_service(config):
    """Create BigQuery service instance"""
    return BigQueryService(
        project_id=config.gcp.project_id,
        dataset_id=config.bigquery.dataset_id
    )


def test_config_loading(config):
    """Test 1: Configuration loading"""
    assert config is not None
    assert config.gcp.project_id is not None, "GCP_PROJECT_ID not set in .env"
    assert config.bigquery.dataset_id is not None, "BIGQUERY_DATASET_ID not set in .env"

    print(f"\n✓ Config loaded successfully")
    print(f"  - GCP Project ID: {config.gcp.project_id}")
    print(f"  - GCP Location: {config.gcp.location}")
    print(f"  - BigQuery Dataset: {config.bigquery.dataset_id}")
    print(f"  - BigQuery Table: {config.bigquery.table_id}")
    print(f"  - Default Commodity: {config.bigquery.commodity}")


def test_bigquery_initialization(bq_service, config):
    """Test 2: BigQuery service initialization"""
    assert bq_service is not None
    assert bq_service.project_id == config.gcp.project_id
    assert bq_service.dataset_id == config.bigquery.dataset_id

    print(f"\n✓ BigQueryService initialized")
    print(f"  - Project ID: {bq_service.project_id}")
    print(f"  - Dataset ID: {bq_service.dataset_id}")


def test_safe_read(bq_service, config):
    """Test 3: Safe read with LIMIT"""
    df = bq_service.test_read_daily_prices(
        commodity=config.bigquery.commodity,
        limit=5
    )

    assert df is not None
    assert len(df) >= 0, "Query should return DataFrame (can be empty)"

    if len(df) > 0:
        # Verify expected columns exist
        expected_cols = ['commodity', 'date', 'open', 'high', 'low', 'close', 'ema', 'volume', 'ingested_at']
        for col in expected_cols:
            assert col in df.columns, f"Column '{col}' missing from daily_prices table"

        # Verify commodity filter worked
        assert (df['commodity'] == config.bigquery.commodity).all(), \
            f"Expected all rows to be '{config.bigquery.commodity}'"

    print(f"\n✓ Query executed successfully")
    print(f"  - Rows fetched: {len(df)}")
    print(f"  - Columns: {', '.join(df.columns.tolist())}")

    if len(df) > 0:
        print(f"\nSample data (first row):")
        print("-" * 60)
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")


def test_date_range_query(bq_service, config):
    """Test 4: Date range query"""
    df = bq_service.get_daily_prices(
        commodity=config.bigquery.commodity,
        start_date="2024-01-01",
        end_date="2024-01-31"
    )

    assert df is not None
    assert len(df) >= 0, "Query should return DataFrame (can be empty)"

    print(f"\n✓ Date range query executed successfully")
    print(f"  - Rows fetched: {len(df)}")

    if len(df) > 0:
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  - Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    else:
        print(f"  ⚠ No data found for date range (this is OK for testing)")


def test_prophet_features(bq_service, config):
    """Test 5: Prophet features query"""
    df = bq_service.get_prophet_features(
        target_date="2024-12-31",
        lookback_days=30,
        commodity=config.bigquery.commodity
    )

    assert df is not None
    assert len(df) >= 0, "Query should return DataFrame (can be empty)"

    if len(df) > 0:
        # Verify Prophet format columns
        assert 'ds' in df.columns, "Prophet requires 'ds' column"
        assert 'y' in df.columns, "Prophet requires 'y' column"

        # Verify data types
        assert df['y'].dtype in ['float64', 'int64'], "'y' column should be numeric"

    print(f"\n✓ Prophet features query executed successfully")
    print(f"  - Rows fetched: {len(df)}")

    if len(df) > 0:
        print(f"  - Columns: {', '.join(df.columns.tolist())}")
        print(f"  - Date range: {df['ds'].min()} to {df['ds'].max()}")
