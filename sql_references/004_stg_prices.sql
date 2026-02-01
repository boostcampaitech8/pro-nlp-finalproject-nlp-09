CREATE TABLE IF NOT EXISTS `db-tutorial1-485202.market.stg_prices` (
  commodity STRING,
  date DATE,
  open FLOAT64,
  high FLOAT64,
  low  FLOAT64,
  close FLOAT64,
  ema FLOAT64,
  volume INT64
);