CREATE TABLE IF NOT EXISTS `db-tutorial1-485202.market.daily_prices` (
  commodity STRING,          -- 'corn' | 'soybean' | 'wheat' | ...
  date DATE,
  open FLOAT64,
  high FLOAT64,
  low  FLOAT64,
  close FLOAT64,
  ema FLOAT64,
  volume INT64,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY commodity;
