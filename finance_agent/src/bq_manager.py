import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "project-5b75bb04-485d-454e-af7"
DATASET = "tilda"
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
