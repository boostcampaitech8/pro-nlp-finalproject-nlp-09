"""
Tests for BigQuery service module

Tests BigQueryService with mocked BigQuery client.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from libs.gcp.bigquery import BigQueryService


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_bigquery_service_initialization(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test BigQueryService initialization"""
    mock_default.return_value = (mock_credentials, None)
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="test_dataset")

    assert service.project_id == "test-project"
    assert service.dataset_id == "test_dataset"
    assert mock_default.called


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_execute_query(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test execute_query method"""
    mock_default.return_value = (mock_credentials, None)

    # Mock query result
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_query_job = Mock()
    mock_query_job.to_dataframe.return_value = mock_df

    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="test_dataset")
    result = service.execute_query("SELECT * FROM table")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    mock_client.query.assert_called_once()


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_get_prophet_features(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test get_prophet_features method"""
    mock_default.return_value = (mock_credentials, None)

    mock_df = pd.DataFrame({"ds": ["2025-01-01"], "y": [100.0]})
    mock_query_job = Mock()
    mock_query_job.to_dataframe.return_value = mock_df

    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="test_dataset")
    result = service.get_prophet_forecast_features(
        target_date="2025-01-20", lookback_days=60, table_id="corn_price"
    )

    assert isinstance(result, pd.DataFrame)
    mock_client.query.assert_called_once()

    # Check query construction
    call_args = mock_client.query.call_args[0][0]
    assert "2025-01-20" in call_args
    assert "corn_price" in call_args


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_get_news_for_prediction(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test get_news_for_prediction method"""
    mock_default.return_value = (mock_credentials, None)

    mock_df = pd.DataFrame({"publish_date": ["2025-01-20"], "title": ["Test"]})
    mock_query_job = Mock()
    mock_query_job.to_dataframe.return_value = mock_df

    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="test_dataset")
    result = service.get_news_for_prediction(target_date="2025-01-20", lookback_days=7)

    assert isinstance(result, pd.DataFrame)

    # Check query includes filter_status = 'T'
    call_args = mock_client.query.call_args[0][0]
    assert "filter_status = 'T'" in call_args


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_invalid_date_format(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test invalid date format raises ValueError"""
    mock_default.return_value = (mock_credentials, None)
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="test_dataset")

    with pytest.raises(ValueError, match="Invalid date format"):
        service.get_prophet_forecast_features(
            target_date="2025/01/20",  # Wrong format
            table_id="corn_price",
        )


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_missing_dataset_id(mock_default, mock_client_class, mock_credentials):
    """Test missing dataset_id raises ValueError"""
    mock_default.return_value = (mock_credentials, None)
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    service = BigQueryService()

    with pytest.raises(ValueError, match="dataset_id"):
        service.get_prophet_forecast_features(target_date="2025-01-20")


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_test_read_daily_prices(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test test_read_daily_prices method (safe query with LIMIT)"""
    mock_default.return_value = (mock_credentials, None)

    # Mock sample data from daily_prices
    mock_df = pd.DataFrame(
        {
            "commodity": ["corn", "corn"],
            "date": ["2025-01-20", "2025-01-19"],
            "close": [450.5, 448.3],
        }
    )
    mock_query_job = Mock()
    mock_query_job.to_dataframe.return_value = mock_df

    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="market")
    result = service.test_read_daily_prices(commodity="corn", limit=10)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    # Check query includes LIMIT and commodity filter
    call_args = mock_client.query.call_args[0][0]
    assert "LIMIT 10" in call_args
    assert "commodity = 'corn'" in call_args
    assert "daily_prices" in call_args


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_get_daily_prices(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test get_daily_prices method with commodity filter"""
    mock_default.return_value = (mock_credentials, None)

    mock_df = pd.DataFrame(
        {
            "commodity": ["corn"] * 3,
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "close": [450.0, 451.0, 452.0],
        }
    )
    mock_query_job = Mock()
    mock_query_job.to_dataframe.return_value = mock_df

    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="market")
    result = service.get_daily_prices(
        commodity="corn", start_date="2025-01-01", end_date="2025-01-03"
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3

    # Check query construction
    call_args = mock_client.query.call_args[0][0]
    assert "commodity = 'corn'" in call_args
    assert "2025-01-01" in call_args
    assert "2025-01-03" in call_args
    assert "daily_prices" in call_args


@patch("libs.gcp.bigquery.bigquery.Client")
@patch("libs.gcp.base.default")
def test_get_prophet_features_with_commodity(
    mock_default, mock_client_class, mock_credentials, mock_env_vars
):
    """Test get_prophet_features with commodity parameter"""
    mock_default.return_value = (mock_credentials, None)

    mock_df = pd.DataFrame(
        {
            "ds": ["2025-01-01", "2025-01-02"],
            "y": [100.0, 101.0],
            "open": [99.0, 100.0],
            "high": [102.0, 103.0],
            "low": [98.0, 99.0],
            "ema": [100.5, 101.5],
            "volume": [1000, 1100],
        }
    )
    mock_query_job = Mock()
    mock_query_job.to_dataframe.return_value = mock_df

    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    mock_client_class.return_value = mock_client

    service = BigQueryService(dataset_id="market")
    result = service.get_prophet_forecast_features(
        target_date="2025-01-20", lookback_days=60, commodity="wheat"
    )

    assert isinstance(result, pd.DataFrame)
    assert "ds" in result.columns
    assert "y" in result.columns

    # Check query includes commodity filter
    call_args = mock_client.query.call_args[0][0]
    assert "commodity = 'wheat'" in call_args
    assert "daily_prices" in call_args
