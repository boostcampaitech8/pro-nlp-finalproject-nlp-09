import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "project-5b75bb04-485d-454e-af7"
DATASET = "tilda"
TEST_COMMODITY_TABLE = "decision_meta"
SENTIMENT_TABLE = "sentiment_result_pos_neg"
TABLE_MAP = {
    "corn": "prophet_corn",
    "soybean": "prophet_soybean",
    "wheat": "prophet_wheat",
}

EFFECT_COLUMN_MAP = {
    "corn": {
        "volume_effect": "Volume_lag5_effect",
        "ema_effect": "EMA_lag2_effect",
    },
    "soybean": {
        "volume_effect": "Volume_lag1_effect",
        "ema_effect": "EMA_lag1_effect",
    },
    "wheat": {
        "volume_effect": "Volume_lag1_effect",
        "ema_effect": "EMA_lag1_effect",
    },
}


def get_performance_data(start_date=None, end_date=None, commodity="corn"):
    client = bigquery.Client(project=PROJECT_ID)
    table = TABLE_MAP.get(commodity, TABLE_MAP["corn"])
    effect_cols = EFFECT_COLUMN_MAP.get(commodity, EFFECT_COLUMN_MAP["corn"])

    query = f"""
        SELECT
            ds AS target_date,
            y_next AS actual_price,
            yhat AS forecast_price,
            yhat_upper,
            yhat_lower,
            direction,
            y_change,
            volatility,
            trend,
            yearly,
            weekly,
            extra_regressors_multiplicative,
            {effect_cols["volume_effect"]} AS Volume_lag_effect,
            {effect_cols["ema_effect"]} AS EMA_lag_effect
        FROM `{PROJECT_ID}.{DATASET}.{table}`
        WHERE ds BETWEEN @start_date AND @end_date
        ORDER BY ds ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    return df


def get_test_commodity_data(
    commodity=None,
    stage=None,
    start_date="2025-10-01",
    end_date="2025-10-31",
):
    """
    decision_meta 테이블에서 의사결정 메타 데이터 조회.
    기본값은 2025년 10월 데이터만 조회합니다.
    """
    client = bigquery.Client(project=PROJECT_ID)

    filters = ["target_date BETWEEN @start_date AND @end_date"]
    query_params = [
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ]

    if commodity:
        filters.append("commodity = @commodity")
        query_params.append(bigquery.ScalarQueryParameter("commodity", "STRING", commodity))

    if stage:
        filters.append("stage = @stage")
        query_params.append(bigquery.ScalarQueryParameter("stage", "STRING", stage))

    where_clause = " AND ".join(filters)

    query = f"""
        SELECT
            ingested_at,
            target_date,
            commodity,
            stage,
            recommendation,
            p_buy,
            p_hold,
            p_sell,
            rationale_short,
            warnings
        FROM `{PROJECT_ID}.{DATASET}.{TEST_COMMODITY_TABLE}`
        WHERE {where_clause}
        ORDER BY target_date ASC, ingested_at DESC
    """

    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    return client.query(query, job_config=job_config).to_dataframe()


def get_sentiment_result_data(start_date=None, end_date=None, keyword=None):
    """
    result_pos_neg 테이블에서 뉴스 감성 집계 데이터를 조회합니다.
    """
    client = bigquery.Client(project=PROJECT_ID)

    filters = ["date BETWEEN @start_date AND @end_date"]
    query_params = [
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ]

    if keyword:
        filters.append("LOWER(keyword) = LOWER(@keyword)")
        query_params.append(bigquery.ScalarQueryParameter("keyword", "STRING", keyword))

    where_clause = " AND ".join(filters)
    query = f"""
        SELECT
            date,
            news_count,
            pos_ratio,
            neg_ratio,
            keyword
        FROM `{PROJECT_ID}.{DATASET}.{SENTIMENT_TABLE}`
        WHERE {where_clause}
        ORDER BY date ASC
    """

    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    return client.query(query, job_config=job_config).to_dataframe()
