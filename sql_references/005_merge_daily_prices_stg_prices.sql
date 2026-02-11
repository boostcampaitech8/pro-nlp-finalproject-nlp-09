MERGE `db-tutorial1-485202.market.daily_prices` T
USING `db-tutorial1-485202.market.stg_prices` S
ON T.commodity = S.commodity AND T.date = S.date
WHEN MATCHED THEN UPDATE SET
  open=S.open, high=S.high, low=S.low, close=S.close, ema=S.ema, volume=S.volume,
  ingested_at=CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
  INSERT (commodity, date, open, high, low, close, ema, volume)
  VALUES (S.commodity, S.date, S.open, S.high, S.low, S.close, S.ema, S.volume);
