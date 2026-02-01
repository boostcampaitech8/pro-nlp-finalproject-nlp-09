-- 시계열 예측 결과 저장 테이블
CREATE TABLE IF NOT EXISTS `db-tutorial1-485202.market.prediction_timeseries` (
    target_date DATE NOT NULL,          -- 예측 기준일
    forecast_value FLOAT64,             -- 예측 가격 (Prophet yhat)
    forecast_direction STRING,          -- 예측 방향 (Up/Down)
    confidence_score FLOAT64,           -- 예측 신뢰도 (%)
    recent_mean_7d FLOAT64,             -- 최근 7일 평균
    all_time_mean FLOAT64,              -- 전체 기간 평균
    trend_analysis STRING,              -- 추세 분석 결과 (Rising/Falling)
    volatility_index FLOAT64,           -- 변동성 지수 (최근 표준편차)
    last_observed_value FLOAT64,        -- 마지막 실제 관측 가격
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() -- 적재 시간
)
PARTITION BY target_date
CLUSTER BY forecast_direction;
