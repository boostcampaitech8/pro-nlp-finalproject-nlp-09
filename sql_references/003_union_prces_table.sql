INSERT INTO `db-tutorial1-485202.market.daily_prices`
(commodity, date, open, high, low, close, ema, volume)
SELECT 'corn', DATE(time), open, high, low, close, EMA, Volume
FROM `db-tutorial1-485202.corns.corn_prices`
UNION ALL
SELECT 'soybean', DATE(time), open, high, low, close, EMA, Volume
FROM `db-tutorial1-485202.soybean.soybean_prices`
UNION ALL
SELECT 'wheat', DATE(time), open, high, low, close, EMA, Volume
FROM `db-tutorial1-485202.wheat.wheat_prices`;